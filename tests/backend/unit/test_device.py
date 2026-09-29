"""Inference device selection: CUDA preferred, CPU fallback, explicit errors when fallback is off."""

from __future__ import annotations

import logging

import pytest

torch = pytest.importorskip("torch")

from traffic_intel.computer_vision.device import DeviceUnavailableError, resolve_device  # noqa: E402


@pytest.fixture
def no_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 0)
    if getattr(torch.backends, "mps", None):
        monkeypatch.setattr(torch.backends.mps, "is_available", lambda: False)


@pytest.fixture
def fake_cuda(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "Fake GPU")


def test_auto_prefers_cuda(fake_cuda: None) -> None:
    info = resolve_device("auto")
    assert (info.device, info.kind, info.name, info.half, info.fell_back) == ("cuda:0", "cuda", "Fake GPU", True, False)
    assert "cuda:0 [Fake GPU" in info.describe() and "fp16" in info.describe()


def test_auto_without_gpu_uses_cpu_fp32(no_cuda: None) -> None:
    info = resolve_device("auto")
    assert info.device == "cpu" and not info.half and not info.fell_back


def test_requested_cuda_falls_back_with_warning(no_cuda: None, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        info = resolve_device("cuda:0")
    assert info.device == "cpu" and info.fell_back and "fell back" in info.describe()
    assert "falling back to CPU" in caplog.text


def test_missing_gpu_index_falls_back(fake_cuda: None) -> None:
    assert resolve_device("cuda:3").device == "cpu"


def test_fallback_can_be_disabled(no_cuda: None) -> None:
    with pytest.raises(DeviceUnavailableError):
        resolve_device("cuda", fallback_to_cpu=False)


def test_half_only_on_cuda(fake_cuda: None) -> None:
    assert resolve_device("cuda", half=False).half is False
    assert resolve_device("cpu", half=True).half is False


def test_unknown_device_rejected() -> None:
    with pytest.raises(ValueError):
        resolve_device("tpu")


@pytest.mark.gpu
@pytest.mark.skipif(not torch.cuda.is_available(), reason="no CUDA device")
def test_real_cuda_device_is_used() -> None:
    info = resolve_device("auto")
    assert info.kind == "cuda" and info.half and info.name
