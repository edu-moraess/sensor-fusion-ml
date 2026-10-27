"""
Environment and runtime audit utilities.

Never installs packages or downloads data. Only inspects the current environment.
"""

from __future__ import annotations

import json
import os
import platform
import sys
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class EnvironmentReport:
    status: str
    timestamp_utc: str
    python: Dict[str, str]
    platform: Dict[str, str]
    packages: Dict[str, Optional[str]]
    torch: Dict[str, Any]
    gpu: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2, default=str)
        return path

    def __str__(self) -> str:
        lines = [
            f"EnvironmentReport [{self.status}]",
            f"UTC: {self.timestamp_utc}",
            f"Python: {self.python.get('version')}",
            f"Platform: {self.platform.get('system')} {self.platform.get('release')}",
            f"Torch: {self.torch.get('version')} | CUDA available: {self.torch.get('cuda_available')}",
            f"GPU: {self.gpu.get('device_name', 'N/A')}",
        ]
        if self.warnings:
            lines.append("Warnings:")
            for w in self.warnings:
                lines.append(f"  - {w}")
        return "\n".join(lines)


def _pkg_version(name: str) -> Optional[str]:
    try:
        import importlib.metadata
        return importlib.metadata.version(name)
    except Exception:
        try:
            mod = __import__(name)
            return getattr(mod, "__version__", "unknown")
        except Exception:
            return None


def audit_environment(
    *,
    require_torch: bool = True,
    require_cuda: bool = False,
) -> EnvironmentReport:
    """
    Inspect the runtime environment without side effects.
    """
    warnings: List[str] = []
    notes: List[str] = []

    python_info = {
        "version": sys.version.split()[0],
        "executable": sys.executable,
        "implementation": platform.python_implementation(),
    }

    platform_info = {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "processor": platform.processor() or "unknown",
    }

    packages = {
        "numpy": _pkg_version("numpy"),
        "scipy": _pkg_version("scipy"),
        "yaml": _pkg_version("PyYAML") or _pkg_version("yaml"),
        "torch": _pkg_version("torch"),
        "cv2": _pkg_version("opencv-python-headless") or _pkg_version("opencv-python"),
        "PIL": _pkg_version("Pillow"),
        "pytest": _pkg_version("pytest"),
    }

    # Torch / CUDA
    torch_info: Dict[str, Any] = {
        "version": packages.get("torch"),
        "cuda_available": False,
        "cuda_version": None,
        "cudnn_version": None,
        "device_count": 0,
    }
    gpu_info: Dict[str, Any] = {
        "device_name": None,
        "memory_total_mb": None,
        "capability": None,
    }

    try:
        import torch
        torch_info["version"] = torch.__version__
        torch_info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            torch_info["cuda_version"] = getattr(torch.version, "cuda", None)
            torch_info["device_count"] = torch.cuda.device_count()
            try:
                torch_info["cudnn_version"] = torch.backends.cudnn.version()
            except Exception:
                pass
            try:
                props = torch.cuda.get_device_properties(0)
                gpu_info["device_name"] = props.name
                gpu_info["memory_total_mb"] = round(props.total_memory / (1024 ** 2))
                gpu_info["capability"] = f"{props.major}.{props.minor}"
            except Exception as e:
                warnings.append(f"Could not query GPU properties: {e}")
        else:
            notes.append("CUDA not available – running on CPU only.")
    except ImportError:
        if require_torch:
            warnings.append("PyTorch is not installed.")
        torch_info["version"] = None

    # Status decision
    status = "PASS"
    if require_torch and packages.get("torch") is None:
        status = "FAIL"
    elif require_cuda and not torch_info.get("cuda_available"):
        status = "BLOCKED" if require_cuda else "PASS_WITH_WARNINGS"
        warnings.append("CUDA required but not available.")
    elif warnings:
        status = "PASS_WITH_WARNINGS"

    # Colab detection (informational)
    if "google.colab" in sys.modules or os.environ.get("COLAB_GPU") is not None:
        notes.append("Running inside Google Colab environment.")
    if os.environ.get("COLAB_GPU") == "1":
        notes.append("Colab GPU runtime flag detected.")

    return EnvironmentReport(
        status=status,
        timestamp_utc=datetime.now(timezone.utc).isoformat(),
        python=python_info,
        platform=platform_info,
        packages=packages,
        torch=torch_info,
        gpu=gpu_info,
        warnings=warnings,
        notes=notes,
    )
