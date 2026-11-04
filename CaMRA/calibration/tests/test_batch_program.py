import pytest
import torch

from sim.batch_program import TableProgram


def make(n=3, t=5, warmup=0.0, dt=0.1):
    tables = torch.arange(n * t * 3, dtype=torch.float32).reshape(n, t, 3) / 100.0
    return TableProgram(tables, dt=dt, warmup_s=warmup), tables.numpy()


def test_per_env_commands_differ():
    prog, _ = make()
    t = torch.zeros(3)
    out = prog.at(t)
    assert out.shape == (3, 3)
    assert not torch.allclose(out[0], out[1])


def test_zero_order_hold_within_interval():
    prog, _ = make()
    a = prog.at(torch.tensor([0.0, 0.0, 0.0]))[0]
    b = prog.at(torch.tensor([0.09, 0.09, 0.09]))[0]
    c = prog.at(torch.tensor([0.1, 0.1, 0.1]))[0]
    assert torch.equal(a, b)
    assert not torch.equal(b, c)


def test_warmup_forces_zero():
    prog, _ = make(warmup=1.0)
    out = prog.at(torch.tensor([0.0, 0.5, 0.99]))
    assert torch.count_nonzero(out) == 0
    after = prog.at(torch.tensor([1.0, 1.0, 1.0]))
    assert torch.count_nonzero(after) > 0


def test_clamps_beyond_table():
    prog, tbl = make(n=2, t=4)
    last = prog.at(torch.tensor([10.0, 10.0]))
    assert torch.allclose(last[0], torch.tensor(tbl[0, -1]))
    assert torch.allclose(last[1], torch.tensor(tbl[1, -1]))


def test_bad_shapes_rejected():
    with pytest.raises(ValueError):
        TableProgram([[[0.0, 0.0]]])
    prog, _ = make(n=2)
    with pytest.raises(ValueError):
        prog.at(torch.zeros(3))
