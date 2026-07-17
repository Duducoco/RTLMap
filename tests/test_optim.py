from types import SimpleNamespace

import torch

from trainer.optim import configure_adamw_with_scheduler


class _Module(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))
        self.trainer = SimpleNamespace(max_steps=-1, estimated_stepping_batches=100)


def test_plateau_scheduler_monitors_joint_validation_loss() -> None:
    configured = configure_adamw_with_scheduler(
        _Module(),
        learning_rate=1e-4,
        weight_decay=1e-5,
        scheduler_type="plateau",
        warmup_steps=100,
        plateau_monitor="val/pair_total_loss",
        plateau_factor=0.5,
        plateau_patience=5,
        min_learning_rate=1e-6,
    )

    scheduler_config = configured["lr_scheduler"]
    scheduler = scheduler_config["scheduler"]
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler_config["monitor"] == "val/pair_total_loss"
    assert scheduler_config["interval"] == "epoch"
    assert scheduler.factor == 0.5
    assert scheduler.patience == 5
    assert scheduler.min_lrs == [1e-6]
