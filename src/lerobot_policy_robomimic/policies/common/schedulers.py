"""Learning rate schedules of robomimic, as LeRobot scheduler configs."""

from dataclasses import dataclass

from lerobot.optim.schedulers import LRSchedulerConfig
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LambdaLR


@LRSchedulerConfig.register_subclass("robomimic_linear")
@dataclass
class RobomimicLinearSchedulerConfig(LRSchedulerConfig):
    """robomimic's "linear" schedule.

    The learning rate falls linearly to `decay_factor` times its initial value over
    `decay_epochs` epochs and then stays there. robomimic updates it once per epoch, so it
    holds for the `steps_per_epoch` optimizer steps of each epoch.
    """

    decay_epochs: int = 100
    decay_factor: float = 0.1
    steps_per_epoch: int = 100
    num_warmup_steps: int | None = None

    def build(self, optimizer: Optimizer, num_training_steps: int) -> LambdaLR:
        def factor(step: int) -> float:
            epoch = min(step // self.steps_per_epoch, self.decay_epochs)
            return 1 + (self.decay_factor - 1) * epoch / self.decay_epochs

        return LambdaLR(optimizer, factor)
