import pytest
import robomimic_reference
import torch

from lerobot_policy_robomimic.schedulers import RobomimicLinearSchedulerConfig


def make_optimizer():
    return torch.optim.AdamW([torch.nn.Parameter(torch.zeros(1))], lr=1e-4)


def step_scheduler(scheduler):
    scheduler.optimizer.step()
    scheduler.step()


def test_linear_schedule_matches_robomimic_stepped_per_epoch():
    steps_per_epoch = 3
    ours = RobomimicLinearSchedulerConfig(decay_epochs=5, decay_factor=0.1, steps_per_epoch=steps_per_epoch)
    ours = ours.build(make_optimizer(), num_training_steps=30)
    theirs = robomimic_reference.linear_lr_scheduler(make_optimizer(), decay_epochs=5, decay_factor=0.1)

    for step in range(30):
        assert ours.get_last_lr()[0] == pytest.approx(theirs.get_last_lr()[0]), f"step {step}"
        step_scheduler(ours)
        if (step + 1) % steps_per_epoch == 0:
            step_scheduler(theirs)


def test_linear_schedule_holds_after_the_decay():
    scheduler = RobomimicLinearSchedulerConfig(decay_epochs=2, steps_per_epoch=1).build(make_optimizer(), 10)
    for _ in range(5):
        step_scheduler(scheduler)

    assert scheduler.get_last_lr()[0] == pytest.approx(1e-5)
