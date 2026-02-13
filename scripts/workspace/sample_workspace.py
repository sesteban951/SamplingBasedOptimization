"""
Sample leaf-body workspaces from a MuJoCo MJCF model.

How to run:
    python -m scripts.workspace.sample_workspace --xml models/g1/g1_21dof.xml --num-samples 1000
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import argparse
import numpy as np
import mujoco


@dataclass(frozen=True)
class SampleConfig:
    """Configuration for sampling leaf-body point clouds."""

    xml_path: Path
    num_samples: int
    seed: int
    max_attempts: int


def _default_output_path(xml_path: Path, collected_at: datetime) -> Path:
    """Build a default output path from model name and collection timestamp."""

    model_name = xml_path.stem
    timestamp = collected_at.strftime("%Y%m%d%H%M")
    return Path("results/pointclouds") / f"{model_name}_{timestamp}.npz"


def _find_leaf_bodies(model: mujoco.MjModel) -> List[int]:
    """Return body ids (excluding world) that have no child bodies."""

    nbody = model.nbody
    parent_ids = model.body_parentid
    child_counts = np.zeros(nbody, dtype=int)
    for body_id in range(1, nbody):
        parent = parent_ids[body_id]
        if parent >= 0:
            child_counts[parent] += 1

    leaf_ids = [body_id for body_id in range(1, nbody) if child_counts[body_id] == 0]
    return leaf_ids


def _sample_qpos(
    rng: np.random.Generator,
    model: mujoco.MjModel,
    qpos0: np.ndarray,
) -> np.ndarray:
    """Sample a qpos within joint limits (hinge/slide) while keeping others at default."""

    qpos = qpos0.copy()
    for joint_id in range(model.njnt):
        joint_type = model.jnt_type[joint_id]
        is_limited = bool(model.jnt_limited[joint_id])
        if not is_limited:
            continue

        if joint_type in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
            qadr = model.jnt_qposadr[joint_id]
            low, high = model.jnt_range[joint_id]
            qpos[qadr] = rng.uniform(low, high)

    return qpos


def _collect_leaf_pointclouds(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    num_samples: int,
    seed: int,
    max_attempts: int,
) -> Tuple[List[str], np.ndarray]:
    """Sample point clouds for each leaf body or its attached sites in the base/body frame.

    Returns:
        leaf_names: list of site or body names for each point cloud.
        points: array with shape (num_targets, num_samples, 3), expressed in base frame.
    """

    leaf_ids = _find_leaf_bodies(model)
    targets: List[Tuple[str, str, int]] = []
    for body_id in leaf_ids:
        site_ids = [i for i in range(model.nsite) if model.site_bodyid[i] == body_id]
        if site_ids:
            for site_id in site_ids:
                site_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, site_id) or f"site_{site_id}"
                targets.append((site_name, "site", site_id))
        else:
            body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or f"body_{body_id}"
            targets.append((body_name, "body", body_id))

    leaf_names = [name for name, _, _ in targets]
    points = np.zeros((len(targets), num_samples, 3), dtype=np.float64)

    qpos0 = model.qpos0.copy()
    rng = np.random.default_rng(seed)

    sample_idx = 0
    attempts = 0
    while sample_idx < num_samples:
        attempts += 1
        if attempts > max_attempts:
            raise RuntimeError(
                f"Exceeded max_attempts={max_attempts} while collecting {num_samples} samples. "
                "Try increasing joint ranges or max_attempts."
            )

        qpos = _sample_qpos(rng, model, qpos0)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        mujoco.mj_collision(model, data)

        # Skip samples with self-collisions (contacts where both bodies are non-world).
        has_self_collision = False
        for k in range(data.ncon):
            contact = data.contact[k]
            b1 = model.geom_bodyid[contact.geom1]
            b2 = model.geom_bodyid[contact.geom2]
            if b1 != 0 and b2 != 0:
                has_self_collision = True
                break
        if has_self_collision:
            continue

        # base/body frame for expressing the points
        base_id = 1  # first non-world body
        p_base = data.xpos[base_id]
        R_base = data.xmat[base_id].reshape(3, 3)

        for target_idx, (_, target_kind, target_id) in enumerate(targets):
            if target_kind == "site":
                p_world = data.site_xpos[target_id]
            else:
                p_world = data.xpos[target_id]
            points[target_idx, sample_idx, :] = R_base.T @ (p_world - p_base)

        sample_idx += 1

    return leaf_names, points


def _save_pointclouds(out_path: Path, leaf_names: List[str], points: np.ndarray) -> None:
    """Save the leaf-body point clouds to disk as a compressed npz file."""

    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, leaf_names=np.array(leaf_names), points=points)


def _parse_args() -> SampleConfig:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Sample leaf-body point clouds from an MJCF model.")
    parser.add_argument("--xml", type=Path, default=Path("models/srb/srb.xml"), help="Path to MJCF file.")
    parser.add_argument("--num-samples", type=int, default=1000, help="Number of samples per leaf body.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for sampling.")
    parser.add_argument(
        "--max-attempts",
        type=int,
        default=100000,
        help="Max sampling attempts before giving up (to handle collision rejection).",
    )
    args = parser.parse_args()

    if args.num_samples < 1:
        raise ValueError("--num-samples must be >= 1")
    if args.max_attempts < args.num_samples:
        raise ValueError("--max-attempts must be >= --num-samples")

    return SampleConfig(
        xml_path=args.xml,
        num_samples=args.num_samples,
        seed=args.seed,
        max_attempts=args.max_attempts,
    )


def main() -> None:
    """Entry point."""

    cfg = _parse_args()
    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    data = mujoco.MjData(model)

    leaf_names, points = _collect_leaf_pointclouds(
        model=model,
        data=data,
        num_samples=cfg.num_samples,
        seed=cfg.seed,
        max_attempts=cfg.max_attempts,
    )

    collected_at = datetime.now()
    out_path = _default_output_path(cfg.xml_path, collected_at)
    _save_pointclouds(out_path, leaf_names, points)
    print(f"Saved {len(leaf_names)} leaf point clouds with {cfg.num_samples} samples each to {out_path}.")


if __name__ == "__main__":
    main()
