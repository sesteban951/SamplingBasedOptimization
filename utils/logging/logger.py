##
#
#  Logger — TensorBoard scalar logging utility
#
##

# standard imports
import os
import time
from dataclasses import dataclass

# tensorboard imports
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
        self.log_path = self._initialize_log_directory(config.experiment_name)

        # tensorboard writer
        self.writer = SummaryWriter(log_dir=self.log_path)

        print(f"Logger initialized. Logging to: [{self.log_path}]")
        print(f"View progress: [tensorboard --logdir=./logs]")


    ################################ CORE ################################

    def log(self, metrics: dict, step: int):
        """
        Log a dictionary of scalar metrics to TensorBoard.

        Args:
            metrics (dict): A dictionary where keys are metric names (str) and values are scalar values (float or int).
            step (int): The current training step or iteration number.
        """
        # update the current step
        self.step = step

        # write to tensorboard at desired frequency
        if step % self.config.log_freq == 0:
            for tag, value in metrics.items():
                self.writer.add_scalar(tag, value, global_step=step)

    def close(self):
        """Flush and close the TensorBoard writer."""
        self.writer.flush()
        self.writer.close()
        print("Logger closed.")
        print(f"View progress: [tensorboard --logdir=./logs]")


    ################################ HELPERS ################################

    @staticmethod
    def _initialize_log_directory(experiment_name):

        # log direcotry
        log_dir = "./logs"

        # get the current time in YYYY-MM-DD_HH-MM-SS format
        timestamp = time.strftime("%Y-%m-%d_%H_%M_%S")

        # full log path
        log_path = os.path.join(log_dir, experiment_name, timestamp)

        # check if path exists (should be rare due to timestamp, but just in case)
        os.makedirs(log_path, exist_ok=True)

        return log_path


#############################################################
# EXAMPLE USAGE
#############################################################

if __name__ == "__main__":

    # config
    log_config = Logger_Config(
        experiment_name = "g1/g1_jump_cem/",
        log_freq        = 2,
    )

    # create logger
    logger = Logger(log_config)

    # simulate an optimization loop
    N = 50
    for i in range(N):

        # fake metrics (plain floats — as they would be in a real loop)
        cost_total    = 10.0 / (i + 1)
        cost_tracking = 6.0  / (i + 1)
        cost_control  = 4.0  / (i + 1)
        pos_err       = 0.5  / (i + 1)
        vel_err       = 0.2  / (i + 1)

        # log
        logger.log({
            "cost/total":     cost_total,
            "cost/tracking":  cost_tracking,
            "cost/control":   cost_control,
            "error/pos_norm": pos_err,
            "error/vel_norm": vel_err,
        }, step=i)

    # close the logger
    logger.close()

    print("Test complete. Run: tensorboard --logdir=./logs")