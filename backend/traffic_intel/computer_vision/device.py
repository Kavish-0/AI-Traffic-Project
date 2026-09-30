"""Inference device selection: prefer CUDA, fall back to CPU, log what is actually used.

``configs/pipeline.yaml -> detector.params``:
    device: auto            # auto | cuda | cuda:N | mps | cpu
    fallback_to_cpu: true   # requested accelerator missing -> CPU with a warning (false -> error)
    half: auto              # fp16: auto = on for CUDA only; true/false to force

Imported only by detector adapters, so processes that never build a detector
(e.g. the API) never import torch.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Literal

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class DeviceInfo:
    device: str                 # value passed to the framework: "cuda:0", "mps", "cpu"
    kind: Literal["cuda", "mps", "cpu"]
    name: str                   # e.g. "NVIDIA GeForce RTX 4060 Laptop GPU"
    half: bool
    torch_version: str
    cuda_version: str | None
    requested: str
    fell_back: bool

    def describe(self) -> str:
        extra = f", CUDA {self.cuda_version}" if self.cuda_version else ""
        fb = f" (requested '{self.requested}', fell back)" if self.fell_back else ""
        return (f"{self.device} [{self.name}{extra}, torch {self.torch_version}, "
                f"{'fp16' if self.half else 'fp32'}]{fb}")


class DeviceUnavailableError(RuntimeError):
    pass


def resolve_device(requested: str = "auto", *, fallback_to_cpu: bool = True,
                   half: bool | str = "auto") -> DeviceInfo:
    import torch

    req = str(requested).strip().lower()
    cuda_ok = torch.cuda.is_available()
    mps_ok = bool(getattr(torch.backends, "mps", None)) and torch.backends.mps.is_available()

    def cpu(fell_back: bool) -> DeviceInfo:
        import platform

        return DeviceInfo("cpu", "cpu", platform.processor() or "CPU", False, torch.__version__, None,
                          requested, fell_back)

    if req == "auto":
        if cuda_ok:
            req = "cuda:0"
        elif mps_ok:
            req = "mps"
        else:
            return _with_half(cpu(False), half)

    if req.startswith("cuda"):
        index = int(req.split(":", 1)[1]) if ":" in req else 0
        if not cuda_ok or index >= torch.cuda.device_count():
            why = ("torch was built without CUDA" if torch.version.cuda is None
                   else "no CUDA device visible" if not cuda_ok else f"cuda:{index} does not exist")
            if not fallback_to_cpu:
                raise DeviceUnavailableError(f"device '{requested}' unavailable: {why}")
            log.warning("Requested device '%s' unavailable (%s) - falling back to CPU", requested, why)
            return _with_half(cpu(True), half)
        info = DeviceInfo(f"cuda:{index}", "cuda", torch.cuda.get_device_name(index), False,
                          torch.__version__, torch.version.cuda, requested, False)
        return _with_half(info, half)

    if req == "mps":
        if not mps_ok:
            if not fallback_to_cpu:
                raise DeviceUnavailableError("device 'mps' unavailable")
            log.warning("Requested device 'mps' unavailable - falling back to CPU")
            return _with_half(cpu(True), half)
        return _with_half(DeviceInfo("mps", "mps", "Apple MPS", False, torch.__version__, None, requested, False),
                          half)

    if req == "cpu":
        return _with_half(cpu(False), half)
    raise ValueError(f"unknown device '{requested}' (use auto, cuda, cuda:N, mps or cpu)")


def _with_half(info: DeviceInfo, half: bool | str) -> DeviceInfo:
    use_half = info.kind == "cuda" if half == "auto" else bool(half) and info.kind == "cuda"
    if half is True and info.kind != "cuda":
        log.info("fp16 requested but device is %s - using fp32", info.kind)
    return DeviceInfo(**{**info.__dict__, "half": use_half})
