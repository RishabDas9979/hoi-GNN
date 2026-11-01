"""Torch-dependent Phase 5 tests. NOT run by the author (no torch in their sandbox):
run `pytest tests/test_phase5.py -v` and report any failure."""
import json

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torch_geometric")

from torch_geometric.loader import DataLoader

from training.fake_data import make_fake_graph, make_fake_split
from training.losses import MaskedBCE, MaskedFocal, build_loss, pos_weight_from_counts
from training.trainer import Trainer
from utils.checkpoint import load_checkpoint, load_model_from_checkpoint
from utils.tiny_model import TinyModel

A, D = 6, 10


# ---------- losses ----------
def test_masked_bce_ignores_masked_entries():
    logits = torch.tensor([[5.0, -5.0]])
    y = torch.tensor([[1.0, 1.0]])           # second label is badly wrong...
    mask = torch.tensor([[1.0, 0.0]])        # ...but masked out
    assert MaskedBCE()(logits, y, mask).item() < 0.01


@pytest.mark.parametrize("loss", [MaskedBCE(), MaskedFocal()])
def test_zero_pairs_gives_zero_loss_and_backward_works(loss):
    lin = torch.nn.Linear(4, A)
    logits = lin(torch.zeros(0, 4))
    out = loss(logits, torch.zeros(0, A), torch.zeros(0, A))
    assert out.item() == 0.0
    out.backward()


def test_focal_downweights_easy_examples():
    easy, hard = torch.tensor([[6.0]]), torch.tensor([[0.0]])
    y, m = torch.ones(1, 1), torch.ones(1, 1)
    f = MaskedFocal(gamma=2.0, alpha=-1)
    assert f(easy, y, m) < f(hard, y, m)


def test_pos_weight_clipping_and_zero_positive_actions():
    w = pos_weight_from_counts([0, 10, 1], [100, 100, 1000], max_weight=20.0)
    assert w[0] == 1.0 and w[1] == pytest.approx(9.0) and w[2] == 20.0


def test_build_loss_names_and_errors():
    assert isinstance(build_loss({"name": "bce"}), MaskedBCE)
    assert isinstance(build_loss({"name": "focal"}), MaskedFocal)
    assert isinstance(build_loss({"name": "weighted_bce"}, [1] * A, [10] * A), MaskedBCE)
    with pytest.raises(ValueError):
        build_loss({"name": "nope"})


# ---------- fake data / batching ----------
def test_pair_index_is_offset_when_batching():
    gs = make_fake_split(8, seed=0)
    gs = [g for g in gs if g.pair_index.size(1) > 0][:2]
    b = next(iter(DataLoader(gs, batch_size=2)))
    assert b.pair_index.max().item() < b.x.size(0)
    assert b.pair_index[:, gs[0].pair_index.size(1):].min().item() >= gs[0].x.size(0)


# ---------- trainer ----------
def _setup(tmp_path, epochs=25, n=60, seed=0):
    tcfg = {"epochs": epochs, "batch_size": 16, "lr": 5e-3, "weight_decay": 0.0, "warmup_epochs": 1,
            "grad_clip": 1.0, "grad_accum": 1, "amp": False, "patience": 0, "rare_below": 5}
    tr = DataLoader(make_fake_split(n, seed), batch_size=16, shuffle=True)
    va = DataLoader(make_fake_split(20, seed + 1), batch_size=16)
    runtime = {"num_actions": A, "node_dim": D, "edge_dim": 9}
    meta = {"model_name": "tiny", "runtime": runtime, "seed": seed, "cfg": {"training": tcfg}}
    torch.manual_seed(seed)
    trainer = Trainer(TinyModel(D, A), MaskedBCE(), tcfg, torch.device("cpu"), tmp_path, meta, [10] * A)
    return trainer, tr, va


def test_training_reduces_loss_and_beats_chance(tmp_path):
    trainer, tr, va = _setup(tmp_path)
    hist = trainer.fit(tr, va)
    assert hist[-1]["train_loss"] < hist[0]["train_loss"]
    assert hist[-1]["val_mAP"] > 0.5       # fake data has a learnable signal


def test_checkpoints_written_and_rebuildable(tmp_path):
    trainer, tr, va = _setup(tmp_path, epochs=2)
    trainer.fit(tr, va)
    assert (tmp_path / "best.pt").exists() and (tmp_path / "last.pt").exists()
    ckpt = load_checkpoint(tmp_path / "best.pt")
    assert ckpt["seed"] == 0 and ckpt["runtime"]["num_actions"] == A and len(ckpt["history"]) >= 1
    model, _ = load_model_from_checkpoint(tmp_path / "best.pt")
    b = next(iter(va))
    assert model(b).shape == (b.pair_index.size(1), A)


def test_early_stopping_stops_before_max_epochs(tmp_path):
    trainer, tr, va = _setup(tmp_path, epochs=50)
    trainer.cfg.update({"lr": 1e-9, "patience": 2, "min_delta": 10.0})  # never "improves"
    hist = trainer.fit(tr, va)
    assert len(hist) <= 4


def test_model_without_parameters_skips_optimizer(tmp_path):
    class Const(torch.nn.Module):
        def forward(self, data):
            return torch.zeros(data.pair_index.size(1), A)

    trainer, tr, va = _setup(tmp_path)
    trainer.model = Const()
    hist = trainer.fit(tr, va)
    assert len(hist) == 1 and (tmp_path / "best.pt").exists()


def test_same_seed_gives_same_result(tmp_path):
    out = []
    for i in range(2):
        trainer, tr, va = _setup(tmp_path / str(i), epochs=3, seed=7)
        out.append(trainer.fit(tr, va)[-1]["val_loss"])
    assert out[0] == pytest.approx(out[1], rel=1e-5)


def test_cli_smoke_end_to_end(tmp_path):
    from training.train import main

    run = main(["--model", "tiny", "--data", "fake", "--seed", "0", "--epochs", "2",
                "--out", str(tmp_path), "--device", "cpu"])
    m = json.loads((run / "metrics.json").read_text())
    assert m["model"] == "tiny" and "mAP" in m["test"] and (run / "history.json").exists()
