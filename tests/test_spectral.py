"""Unit tests for spectral preprocessing."""
import numpy as np
import pytest
import torch

from eigenfraud.spectral.azimuthal import azimuthal_average, spectrum_to_1d_input
from eigenfraud.spectral.spectrum2d import compute_log_spectrum_2d


class TestAzimuthalAverage:
    def test_output_length_square(self):
        img = np.random.rand(256, 256).astype(np.float32)
        bins, psd = azimuthal_average(img)
        assert len(bins) == len(psd)
        assert len(bins) == 256 // 2 + 1  # r_max + 1

    def test_output_length_rectangular(self):
        img = np.random.rand(128, 256).astype(np.float32)
        bins, psd = azimuthal_average(img)
        assert len(bins) == 128 // 2 + 1  # truncated at min(H,W)//2

    def test_dc_suppressed(self):
        # A constant image has zero variance; DC bin should be near zero after mean subtraction
        img = np.ones((64, 64), dtype=np.float32) * 128.0
        _, psd = azimuthal_average(img)
        assert psd[0] < 1e-6

    def test_nonnegative(self):
        img = np.random.rand(64, 64).astype(np.float32)
        _, psd = azimuthal_average(img)
        assert np.all(psd >= 0)

    def test_known_frequency_peak(self):
        # A horizontal sinusoid at f=0.25 cycles/pixel should produce a peak there
        size = 128
        x = np.arange(size)
        freq = 0.25  # cycles/pixel
        img = np.sin(2 * np.pi * freq * x)[None, :] * np.ones((size, 1))
        img = img.astype(np.float32)
        bins, psd = azimuthal_average(img)
        # Peak bin is at freq * size = 32
        peak_bin = int(round(freq * size))
        # The peak bin should have above-average power
        assert psd[peak_bin] > psd.mean()


class TestSpectrumTo1DInput:
    def test_output_shape_2d(self):
        img = np.random.rand(256, 256).astype(np.float32)
        vec = spectrum_to_1d_input(img)
        assert vec.shape == (256 // 2 + 1,)
        assert vec.dtype == np.float32

    def test_output_shape_3d(self):
        img = np.random.rand(256, 256, 3).astype(np.float32)
        vec = spectrum_to_1d_input(img)
        assert vec.shape == (256 // 2 + 1,)

    def test_normalized_range(self):
        img = np.random.rand(64, 64).astype(np.float32)
        vec = spectrum_to_1d_input(img)
        assert vec.min() >= 0.0
        assert vec.max() <= 1.0 + 1e-6


class TestComputeLogSpectrum2D:
    def test_output_shape(self):
        img = torch.rand(3, 256, 256)
        out = compute_log_spectrum_2d(img, output_size=224)
        assert out.shape == (3, 224, 224)

    def test_output_range(self):
        img = torch.rand(3, 256, 256)
        out = compute_log_spectrum_2d(img, output_size=224)
        assert out.min().item() >= -1e-6
        assert out.max().item() <= 1.0 + 1e-6

    def test_constant_image_is_zero(self):
        # A constant image — after mean subtraction — has all-zero FFT → log(1+0)=0
        img = torch.ones(3, 64, 64) * 0.5
        out = compute_log_spectrum_2d(img, output_size=64)
        assert out.abs().max().item() < 1e-5

    def test_grad_flows(self):
        img = torch.rand(3, 64, 64, requires_grad=True)
        out = compute_log_spectrum_2d(img, output_size=64)
        out.sum().backward()
        assert img.grad is not None
