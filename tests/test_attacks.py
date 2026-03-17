"""Tests for spatial and frequency PGD attacks."""
import torch
import torch.nn as nn
import pytest

from eigenfraud.attacks.pgd_spatial import pgd_spatial
from eigenfraud.attacks.pgd_freq import pgd_freq
from eigenfraud.attacks.bandlimited import BandMask, pgd_freq_bandlimited


class TinyModel(nn.Module):
    """Minimal model returning a single logit per image."""
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(3 * 16 * 16, 1)

    def forward(self, x):
        return self.fc(x.flatten(1))


def make_batch(B=4, C=3, H=16, W=16):
    X = torch.rand(B, C, H, W)
    y = torch.randint(0, 2, (B,))
    return X, y


class TestPGDSpatial:
    def test_output_shape(self):
        model = TinyModel()
        X, y = make_batch()
        adv = pgd_spatial(model, X, y, eps=8/255, alpha=2/255, num_iter=5)
        assert adv.shape == X.shape

    def test_output_in_range(self):
        model = TinyModel()
        X, y = make_batch()
        adv = pgd_spatial(model, X, y, eps=8/255, alpha=2/255, num_iter=5)
        assert adv.min().item() >= 0.0
        assert adv.max().item() <= 1.0

    def test_linf_constraint(self):
        model = TinyModel()
        X, y = make_batch()
        eps = 8 / 255
        adv = pgd_spatial(model, X, y, eps=eps, alpha=2/255, num_iter=10)
        delta = (adv - X).abs()
        assert delta.max().item() <= eps + 1e-6


class TestPGDFreq:
    def test_output_shape(self):
        model = TinyModel()
        X, y = make_batch()
        adv = pgd_freq(model, X, y, eps_freq=0.05, alpha_freq=0.005, num_iter=5)
        assert adv.shape == X.shape

    def test_output_in_range(self):
        model = TinyModel()
        X, y = make_batch()
        adv = pgd_freq(model, X, y, eps_freq=0.05, alpha_freq=0.005, num_iter=5)
        assert adv.min().item() >= 0.0
        assert adv.max().item() <= 1.0

    def test_spatial_eps_projection(self):
        model = TinyModel()
        X, y = make_batch()
        spatial_eps = 4 / 255
        adv = pgd_freq(model, X, y, eps_freq=0.05, alpha_freq=0.005,
                       num_iter=5, spatial_eps=spatial_eps)
        assert adv.min().item() >= 0.0
        assert adv.max().item() <= 1.0


class TestBandMask:
    def test_mask_shape(self):
        bm = BandMask(0.1, 0.4)
        mask = bm.make_mask(16, 16)
        assert mask.shape == (16, 16)

    def test_dc_excluded_when_low_cutoff_positive(self):
        bm = BandMask(0.1, 0.5)
        mask = bm.make_mask(16, 16)
        # DC is at index (0, 0) in fftfreq-ordered space; radius = 0 < 0.1
        assert mask[0, 0].item() == 0.0

    def test_all_ones_when_full_range(self):
        bm = BandMask(0.0, 1.0)
        mask = bm.make_mask(16, 16)
        # All frequencies within unit circle should be included;
        # corners may be outside r=1 but that's fine
        assert mask[0, 0].item() == 1.0

    def test_bandlimited_attack_output_shape(self):
        model = TinyModel()
        X, y = make_batch()
        band = BandMask(0.1, 0.5)
        adv = pgd_freq_bandlimited(model, X, y, band, eps_freq=0.05,
                                   alpha_freq=0.005, num_iter=3)
        assert adv.shape == X.shape
        assert adv.min().item() >= 0.0
        assert adv.max().item() <= 1.0
