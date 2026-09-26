import numpy as np
import torch
from ubiomark import models


def test_seeded_model_initialization_ignores_prior_rng_and_preserves_stream():
    torch.manual_seed(919)
    before = torch.get_rng_state().clone()
    a = models.seeded_node_model(models.MLP, 3, dropout=0.0, seed=42)
    assert torch.equal(before, torch.get_rng_state())
    torch.rand(100)
    b = models.seeded_node_model(models.MLP, 3, dropout=0.0, seed=42)
    c = models.seeded_node_model(models.MLP, 3, dropout=0.0, seed=43)
    av = torch.cat([p.detach().flatten() for p in a.parameters()])
    bv = torch.cat([p.detach().flatten() for p in b.parameters()])
    cv = torch.cat([p.detach().flatten() for p in c.parameters()])
    assert torch.equal(av, bv) and not torch.equal(av, cv)


def test_seeded_training_reproducible_after_different_global_rng_states():
    x = torch.tensor(np.arange(24).reshape(8, 3) / 24, dtype=torch.float32)
    y = np.array([0, 1] * 4)
    mask = np.ones(8, bool)
    a = models.seeded_node_model(models.MLP, 3, dropout=0.0, seed=23)
    pa = models.train_node_model(a, x, None, y, mask, epochs=2, seed=23)
    torch.rand(120)
    b = models.seeded_node_model(models.MLP, 3, dropout=0.0, seed=23)
    pb = models.train_node_model(b, x, None, y, mask, epochs=2, seed=23)
    assert np.array_equal(pa, pb)


def test_cnn_seeded_initialization_repeatable_despite_global_rng():
    x = np.random.default_rng(0).normal(size=(8, 12)).astype(np.float32)
    y = np.array([0, 1] * 4)
    a = models.train_cnn(x, y, epochs=2, seed=7, batch=4)
    pa = models.predict_cnn(a, x)
    torch.rand(100)
    b = models.train_cnn(x, y, epochs=2, seed=7, batch=4)
    pb = models.predict_cnn(b, x)
    assert np.array_equal(pa, pb)
