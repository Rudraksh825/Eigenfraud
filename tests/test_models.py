"""Smoke tests for model forward passes and gradient flow."""
import torch

from eigenfraud.models.cnn1d import SpectralCNN1D
from eigenfraud.models.resnet2d import SpectralResNet2D


class TestSpectralCNN1D:
    def test_output_shape(self):
        model = SpectralCNN1D(input_dim=128)
        x = torch.rand(4, 1, 128)
        out = model(x)
        assert out.shape == (4, 1)

    def test_gradients_flow(self):
        model = SpectralCNN1D(input_dim=128)
        x = torch.rand(2, 1, 128)
        loss = model(x).sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_different_input_dim(self):
        model = SpectralCNN1D(input_dim=64)
        x = torch.rand(2, 1, 64)
        out = model(x)
        assert out.shape == (2, 1)


class TestSpectralResNet2D:
    def test_output_shape(self):
        model = SpectralResNet2D(pretrained=False)
        x = torch.rand(2, 3, 224, 224)
        out = model(x)
        assert out.shape == (2, 1)

    def test_gradients_flow(self):
        model = SpectralResNet2D(pretrained=False)
        x = torch.rand(2, 3, 224, 224)
        loss = model(x).sum()
        loss.backward()
        for p in model.parameters():
            assert p.grad is not None

    def test_layer1_stride_modified(self):
        model = SpectralResNet2D(pretrained=False)
        # The first conv in layer1[0] should have stride (1,1), not (2,2)
        layer1_block = model.backbone.layer1[0]
        assert layer1_block.conv1.stride == (1, 1)
        if layer1_block.downsample is not None:
            assert layer1_block.downsample[0].stride == (1, 1)
