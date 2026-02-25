##
#
#  Logger — TensorBoard scalar logging utility
#
##

import os
import time
import numpy as np
import jax.numpy as jnp
from dataclasses import dataclass, field
from typing import Dict, Optional, Union
from tensorboardX import SummaryWriter


#############################################################
# LOGGER CONFIG
#############################################################

@dataclass
class Logger_Config:

    # experiment name
    experiment_name: str

    # log every N steps (set to 1 to log every step)
    log_freq: int = 2

    # print every N steps (only active if verbose=True)
    print_freq: int = 2


#############################################################
# LOGGER CLASS
#############################################################

class Logger:
    """
    Lightweight TensorBoard scalar logger for MJX 
    sampling based optimization loops.
    """

    def __init__(self, config: Logger_Config):

        # logger config
        self.config = config

        # initialize current step
        self.step = 0

        # build the full log path
        self._initialize_log_directory(config.experiment_name)

        # tensorboard writer
        self.writer = SummaryWriter(log_dir=self.log_path)

        # internal scalar buffer: tag -> list of (step, value) for optional inspection
        self._history: Dict[str, list] = {}

        # timing
        self._t_start = time.time()
        self._t_last  = self._t_start

        print(f"Logger initialized. TensorBoard log dir: [{self.log_path}]")
        print(f"  Launch with: tensorboard --logdir={config.log_dir}")

    # ------------------------------------------------------------------
    # PUBLIC API
    # ------------------------------------------------------------------

    def log(self, scalars: Dict[str, Union[float, np.ndarray, "jnp.ndarray"]],
                  step: Optional[int] = None):
        """
        Log a dictionary of scalars. Arrays are auto-reduced to their mean.

        Args:
            scalars: dict mapping tag -> value.
                     Value can be:
                       - Python float / int
                       - numpy array (any shape) — mean is taken
                       - JAX array (any shape)   — transferred to host and mean is taken
            step:    global step index. If None, uses internal counter.
        """
        # resolve step
        if step is None:
            step = self.step
        self.step = step

        # check log frequency
        if step % self.config.log_freq != 0:
            return

        # timing
        t_now     = time.time()
        t_elapsed = t_now - self._t_start
        t_step    = t_now - self._t_last
        self._t_last = t_now

        # reduce and write each scalar
        reduced: Dict[str, float] = {}
        for tag, value in scalars.items():
            scalar = self._to_scalar(value)
            reduced[tag] = scalar
            self.writer.add_scalar(tag, scalar, global_step=step)

            # store history
            if tag not in self._history:
                self._history[tag] = []
            self._history[tag].append((step, scalar))

        # optionally log timing info too
        self.writer.add_scalar("_timing/step_time_s", t_step,    global_step=step)
        self.writer.add_scalar("_timing/elapsed_s",   t_elapsed, global_step=step)

        # print to stdout
        if self.config.verbose and (step % self.config.print_freq == 0):
            self._print(step, reduced, t_step, t_elapsed)


    def log_config(self, config_dict: dict):
        """
        Log a flat dictionary of hyperparameters to TensorBoard's HParams tab.

        Args:
            config_dict: flat dict of param_name -> value (str, int, float, bool)
        """
        # tensorboard hparams requires metric dict too — use empty placeholder
        self.writer.add_hparams(config_dict, metric_dict={})
        print(f"Logged config: {config_dict}")


    def get_history(self, tag: str):
        """
        Retrieve the logged history for a given tag.

        Args:
            tag: scalar tag name (e.g. "cost/total")
        Returns:
            list of (step, value) tuples, or empty list if tag not found
        """
        return self._history.get(tag, [])


    def close(self):
        """Flush and close the TensorBoard writer."""
        self.writer.flush()
        self.writer.close()
        print(f"Logger closed. Total elapsed: {time.time() - self._t_start:.1f}s")


    # ------------------------------------------------------------------
    # PRIVATE HELPERS
    # ------------------------------------------------------------------

    @staticmethod
    def _initialize_log_directory(experiment_name):

        # log direcotry
        log_dir = "./logs"

        # get the current time in YYYY-MM-DD_HH-MM-SS format
        timestamp = time.strftime("%Y-%m-%d_%H:%M:%S")

        # full log path
        log_path = log_dir + "/" + experiment_name + "_" + timestamp

        print(log_path)

        exit(0)

        # # create a subdirectory for this experiment
        # self.log_path = os.path.join(log_dir, f"{experiment_name}_{timestamp}")
        # os.makedirs(self.log_path, exist_ok=True)

    @staticmethod
    def _to_scalar(value) -> float:
        """
        Reduce any array-like value to a single Python float (mean over all elements).
        Handles JAX arrays, numpy arrays, and plain scalars.
        """
        # JAX array -> numpy first (pulls off device)
        if isinstance(value, jnp.ndarray):
            value = np.array(value)

        # numpy array -> take mean
        if isinstance(value, np.ndarray):
            return float(np.mean(value))

        # plain scalar
        return float(value)


    @staticmethod
    def _print(step: int, scalars: Dict[str, float], t_step: float, t_elapsed: float):
        """Pretty-print scalars to stdout."""
        parts = [f"step {step:>6d}"]
        for tag, val in scalars.items():
            parts.append(f"{tag}: {val:.4f}")
        parts.append(f"({t_step:.3f}s/step, {t_elapsed:.1f}s elapsed)")
        print(" | ".join(parts))


#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    import jax
    import jax.numpy as jnp

    # config
    log_config = Logger_Config(
        experiment_name= "g1/g1_jump_cem",
        log_freq       = 5,
        print_freq     = 5,
    )

    # logger
    logger = Logger(log_config)


