"""
Visualize leaf-body workspace point clouds around a MuJoCo model.

How to run:
    python -m scripts.workspace.visualize_pointclouds --xml models/g1/g1_21dof.xml
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import argparse
import time

import numpy as np
import mujoco
import mujoco.viewer


@dataclass(frozen=True)
class ViewerConfig:
    """Configuration for workspace visualization."""

    xml_path: Path
    points_path: Path
    leaf: str | None
    point_radius: float
    max_points_per_leaf: int | None


def _load_pointcloud(points_path: Path) -> Tuple[List[str], np.ndarray]:
    """Load leaf names and points from a compressed npz file."""

    data = np.load(points_path, allow_pickle=True)
    if "points" not in data:
        raise ValueError(f"Pointcloud file {points_path} missing 'points' array.")
    points = np.asarray(data["points"], dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"Expected points shape (num_leaf, num_samples, 3). Got {points.shape}.")

    leaf_names = data.get("leaf_names", None)
    if leaf_names is None:
        leaf_names = np.array([f"leaf_{i}" for i in range(points.shape[0])])
    leaf_names = [str(name) for name in leaf_names.tolist()]

    return leaf_names, points


def _find_latest_pointcloud(points_dir: Path = Path("results/pointclouds")) -> Path:
    """Find the most recently modified pointcloud npz file in the default results folder."""

    if not points_dir.exists():
        raise ValueError(
            f"Default pointcloud folder '{points_dir}' does not exist. "
            "Generate a pointcloud first or pass --points explicitly."
        )

    candidates = [p for p in points_dir.glob("*.npz") if p.is_file()]
    if not candidates:
        raise ValueError(
            f"No pointcloud files found in '{points_dir}'. "
            "Generate a pointcloud first or pass --points explicitly."
        )

    return max(candidates, key=lambda p: p.stat().st_mtime)


def _select_leaves(
    leaf_names: Sequence[str],
    points: np.ndarray,
    leaf: str | None,
) -> Tuple[List[str], np.ndarray]:
    """Select leaf point clouds by name or index string."""

    if leaf is None:
        return list(leaf_names), points

    if leaf.isdigit():
        idx = int(leaf)
        if idx < 0 or idx >= len(leaf_names):
            raise ValueError(f"Leaf index {idx} out of range [0, {len(leaf_names) - 1}].")
        return [leaf_names[idx]], points[idx : idx + 1]

    if leaf not in leaf_names:
        raise ValueError(f"Leaf '{leaf}' not found. Available: {leaf_names}")

    idx = leaf_names.index(leaf)
    return [leaf_names[idx]], points[idx : idx + 1]


def _add_point_geom(
    scene: mujoco.MjvScene,
    pos: np.ndarray,
    rgba: np.ndarray,
    radius: float,
) -> None:
    """Add a small sphere geom to the scene."""

    if scene.ngeom >= scene.maxgeom:
        return

    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, 0.0, 0.0]),
        pos,
        np.eye(3).reshape(9),
        rgba,
    )
    geom.category = int(mujoco.mjtCatBit.mjCAT_DECOR)
    scene.ngeom += 1


def _build_point_geoms(
    scene: mujoco.MjvScene,
    points_body: np.ndarray,
    base_pos: np.ndarray,
    base_rot: np.ndarray,
    rgba: np.ndarray,
    radius: float,
    max_points: int | None,
) -> int:
    """Populate the scene with point geoms for a point set.

    Returns:
        Number of geoms added.
    """

    if max_points is not None and points_body.shape[0] > max_points:
        indices = np.linspace(0, points_body.shape[0] - 1, max_points, dtype=int)
        points_body = points_body[indices]

    before = scene.ngeom
    for p_body in points_body:
        p_world = base_pos + base_rot @ p_body
        _add_point_geom(scene, p_world, rgba, radius)
    return scene.ngeom - before


def _parse_args() -> ViewerConfig:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(description="Visualize leaf-body point clouds.")
    parser.add_argument("--xml", type=Path, default=Path("models/srb/srb.xml"), help="MJCF model path.")
    parser.add_argument(
        "--points",
        type=Path,
        default=None,
        help="Path to pointcloud npz file. If omitted, uses newest file in results/pointclouds.",
    )
    parser.add_argument("--leaf", type=str, default=None, help="Leaf name or index to visualize.")
    parser.add_argument("--point-radius", type=float, default=0.01, help="Point radius (meters).")
    parser.add_argument(
        "--max-points",
        type=int,
        default=2000,
        help="Max points per leaf to render (downsample if needed).",
    )
    args = parser.parse_args()

    if args.point_radius <= 0.0:
        raise ValueError("--point-radius must be > 0")
    if args.max_points is not None and args.max_points < 1:
        raise ValueError("--max-points must be >= 1")

    return ViewerConfig(
        xml_path=args.xml,
        points_path=args.points if args.points is not None else _find_latest_pointcloud(),
        leaf=args.leaf,
        point_radius=args.point_radius,
        max_points_per_leaf=args.max_points,
    )


def main() -> None:
    """Entry point for workspace visualization."""

    cfg = _parse_args()
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    data = mujoco.MjData(model)

    print(f"Using pointcloud file: {cfg.points_path}")
    leaf_names, points_body = _load_pointcloud(cfg.points_path)
    leaf_names, points_body = _select_leaves(leaf_names, points_body, cfg.leaf)
    print("Displaying pointclouds for leaves:")
    for name in leaf_names:
        print(f"  - {name}")

    viewer = mujoco.viewer.launch_passive(model, data)

    palette = np.array(
        [
            [0.9, 0.2, 0.2, 0.8],
            [0.2, 0.7, 0.2, 0.8],
            [0.2, 0.4, 0.9, 0.8],
            [0.9, 0.6, 0.2, 0.8],
            [0.7, 0.2, 0.8, 0.8],
            [0.2, 0.8, 0.7, 0.8],
        ],
        dtype=np.float64,
    )

    try:
        while viewer.is_running():
            if viewer.user_scn is not None:
                viewer.user_scn.ngeom = 0
                base_id = 1  # first non-world body
                base_pos = data.xpos[base_id].copy()
                base_rot = data.xmat[base_id].reshape(3, 3).copy()

                for idx, (name, pts_body) in enumerate(zip(leaf_names, points_body)):
                    color = palette[idx % len(palette)]
                    added = _build_point_geoms(
                        viewer.user_scn,
                        pts_body,
                        base_pos,
                        base_rot,
                        color,
                        cfg.point_radius,
                        cfg.max_points_per_leaf,
                    )
                    if viewer.user_scn.ngeom >= viewer.user_scn.maxgeom:
                        print(
                            f"Warning: reached mujoco.mjMAXGEOM limit. "
                            f"Rendered {added} geoms for leaf '{name}'."
                        )
                        break
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nClosed visualization.")
    finally:
        viewer.close()


if __name__ == "__main__":
    main()
