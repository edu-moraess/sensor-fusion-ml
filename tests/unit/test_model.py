"""Unit tests for MinimalFusionMLP – gradients, shapes, missing modalities."""

import torch
import pytest

from sensor_fusion_ml.fusion.minimal import MinimalFusionMLP


def test_forward_all_modalities():
    model = MinimalFusionMLP(position_dim=2, hidden_dim=64, trajectory_horizon=5)
    B = 4
    rgb = torch.randn(B, 3, 32, 32)
    points = torch.randn(B, 64, 3)
    gnss = torch.randn(B, 2)
    imu = torch.randn(B, 6)
    out = model(rgb=rgb, points=points, gnss=gnss, imu=imu)
    assert out["position"].shape == (B, 2)
    assert out["velocity"].shape == (B, 2)
    assert out["trajectory"].shape == (B, 5, 2)


def test_missing_modalities():
    model = MinimalFusionMLP(position_dim=2, trajectory_horizon=0)
    B = 2
    # Only GNSS
    out = model(gnss=torch.randn(B, 2))
    assert out["position"].shape == (B, 2)
    assert "trajectory" not in out


def test_gradient_flow():
    model = MinimalFusionMLP(position_dim=2, hidden_dim=32, trajectory_horizon=3)
    rgb = torch.randn(2, 3, 16, 16, requires_grad=True)
    out = model(rgb=rgb)
    loss = out["position"].sum() + out["velocity"].sum() + out["trajectory"].sum()
    loss.backward()
    assert rgb.grad is not None
    # At least some parameters received gradient
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert len(grads) > 0


def test_checkpoint_roundtrip(tmp_path):
    model = MinimalFusionMLP(position_dim=2, hidden_dim=32, dropout=0.0)
    model.eval()
    path = tmp_path / "ckpt.pt"
    torch.save({"model_state": model.state_dict(), "meta": {"version": "0.1.0"}}, path)
    loaded = torch.load(path, map_location="cpu", weights_only=True)
    model2 = MinimalFusionMLP(position_dim=2, hidden_dim=32, dropout=0.0)
    model2.load_state_dict(loaded["model_state"])
    model2.eval()
    # Same output on same input
    x = torch.randn(1, 3, 16, 16)
    with torch.no_grad():
        o1 = model(rgb=x)["position"]
        o2 = model2(rgb=x)["position"]
    assert torch.allclose(o1, o2, atol=1e-6)
