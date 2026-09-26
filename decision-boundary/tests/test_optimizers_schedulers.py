import pytest
import torch
from torch import nn

from playground.config import RunConfig
from playground.train.optimizers import SAM, build_optimizer
from playground.train.schedulers import WarmupCosineLR, build_scheduler


def _model() -> nn.Module:
    return nn.Linear(4, 4)


@pytest.mark.parametrize("name", ["sgd", "sgd_momentum", "adam", "adamw", "sam"])
def test_build_optimizer_dispatches(name: str) -> None:
    cfg = RunConfig(optimizer=name)
    opt = build_optimizer(_model(), cfg)
    assert opt is not None
    if name == "sam":
        assert isinstance(opt, SAM)


def test_build_optimizer_unknown_raises() -> None:
    cfg = RunConfig(optimizer="foo")
    with pytest.raises(ValueError, match="unknown optimizer"):
        build_optimizer(_model(), cfg)


def test_sam_rejects_nonpositive_rho() -> None:
    with pytest.raises(ValueError, match="rho must be > 0"):
        SAM(_model().parameters(), base_optimizer_cls=torch.optim.SGD, rho=0.0, lr=0.1)


def test_sam_step_raises() -> None:
    opt = SAM(_model().parameters(), base_optimizer_cls=torch.optim.SGD, rho=0.05, lr=0.1)
    with pytest.raises(RuntimeError, match="first_step"):
        opt.step()


@pytest.mark.parametrize("name", ["constant", "cosine", "step", "warmup_cosine"])
def test_build_scheduler_all_branches(name: str) -> None:
    cfg = RunConfig(lr_schedule=name, epochs=20)
    opt = torch.optim.SGD(_model().parameters(), lr=0.1)
    sched = build_scheduler(opt, cfg)
    if name == "constant":
        assert sched is None
    else:
        assert sched is not None


def test_build_scheduler_unknown_raises() -> None:
    cfg = RunConfig(lr_schedule="foo", epochs=20)
    opt = torch.optim.SGD(_model().parameters(), lr=0.1)
    with pytest.raises(ValueError, match="unknown lr_schedule"):
        build_scheduler(opt, cfg)


def test_warmup_cosine_lr_starts_low_and_decays() -> None:
    opt = torch.optim.SGD(_model().parameters(), lr=1.0)
    sched = WarmupCosineLR(opt, warmup_epochs=5, total_epochs=20)
    lrs = []
    for _ in range(20):
        lrs.append(opt.param_groups[0]["lr"])
        sched.step()
    # First step should be tiny (warmup), peak around epoch 5, then decay.
    assert lrs[0] < lrs[5]
    assert lrs[-1] < lrs[5]
