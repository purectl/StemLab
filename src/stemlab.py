#!/usr/bin/env python3
"""
Stemlab v1.1 - BETA 
maintained at: https://github.com/purectl/StemLab
License: GPL
"""

import subprocess
import sys
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Literal, Dict, Any, List
from pathlib import Path
import platform
import datetime
import math
import tempfile
import traceback
import shutil
import json
import re
import gc
import os


def _get_required_packages():
    return [
        ("torch",            "torch"),
        ("demucs",           "demucs"),
        ("audio_separator",  "audio-separator"),
        ("soundfile",        "soundfile"),
        ("numpy",            "numpy"),
        ("onnxruntime",      "onnxruntime"),
        ("pyqt6",            "pyqt6"),
    ]


def _check_missing_packages():
    missing = []
    for pkg, pip_name in _get_required_packages():
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pip_name)
    return missing


def _detect_python_for_venv() -> str:
    for candidate in ("python3.11", "python3", "python"):
        try:
            r = subprocess.run([candidate, "--version"],
                               capture_output=True, text=True)
            if r.returncode == 0:
                return candidate
        except FileNotFoundError:
            continue
    return sys.executable


def _venv_dir() -> Path:
    return Path(__file__).parent / "stemlab_venv"


def _all_pip_packages():
    return [pip_name for _, pip_name in _get_required_packages()]


def _ensure_deps_unix():
    venv = _venv_dir()
    python_cmd = _detect_python_for_venv()
    if not venv.exists():
        print(f"[StemLab] Creating venv at {venv} using {python_cmd}...")
        r = subprocess.run([python_cmd, "-m", "venv", str(venv)])
        if r.returncode != 0:
            print("[StemLab] venv creation failed — falling back to system pip")
            _ensure_deps_windows()
            return
    pip = venv / "bin" / "pip"
    pkgs = _all_pip_packages()
    print(f"[StemLab] Installing packages into venv: {', '.join(pkgs)}")
    subprocess.run([str(pip), "install", "--upgrade", "pip"],
                   capture_output=True)
    r = subprocess.run([str(pip), "install"] + pkgs)
    if r.returncode != 0:
        print("[StemLab] Some packages failed to install — continuing anyway")
    venv_python = venv / "bin" / "python"
    if venv_python.exists():
        print("[StemLab] Re-launching inside venv...")
        os.execv(str(venv_python), [str(venv_python)] + sys.argv)


def _ensure_deps_windows():
    pkgs = _all_pip_packages()
    print(f"[StemLab] Installing packages: {', '.join(pkgs)}")
    r = subprocess.run(
        [sys.executable, "-m", "pip", "install", "--break-system-packages"] + pkgs)
    if r.returncode != 0:
        subprocess.run([sys.executable, "-m", "pip", "install"] + pkgs)


def ensure_dependencies():
    system = platform.system()
    if system in ("Linux", "Darwin"):
        in_any_venv = sys.prefix != sys.base_prefix
        if in_any_venv:
            missing = _check_missing_packages()
            if missing:
                print(
                    f"[StemLab] Installing missing packages into current venv: {', '.join(missing)}")
                subprocess.run(
                    [sys.executable, "-m", "pip", "install"] + missing)
            return
        _ensure_deps_unix()
    else:
        missing = _check_missing_packages()
        if missing:
            _ensure_deps_windows()


try:
    import tomllib as _toml_lib
    _TOML_OK = True
except ImportError:
    try:
        import tomli as _toml_lib
        _TOML_OK = True
    except ImportError:
        _TOML_OK = False

os.environ.setdefault("QT_STYLE_OVERRIDE", "")
os.environ["QT_LOGGING_RULES"] = "qt.qpa.services=false"


def _app_data_dir() -> Path:
    app_name = "StemLab"
    system = platform.system()
    try:
        if system == "Windows":
            base = Path(os.environ.get(
                "APPDATA", Path.home() / "AppData" / "Roaming"))
            d = base / app_name
        elif system == "Darwin":
            d = Path.home() / "Library" / "Application Support" / app_name
        else:
            xdg = os.environ.get("XDG_DATA_HOME", "")
            base = Path(xdg) if xdg else (Path.home() / ".local" / "share")
            d = base / app_name
        d.mkdir(parents=True, exist_ok=True)
        return d
    except Exception:
        return Path(__file__).parent


def _models_dir() -> Path:
    d = _app_data_dir() / "models"
    d.mkdir(parents=True, exist_ok=True)
    return d


MODELS_JSON = _app_data_dir() / "models.json"

BUILTIN_MODELS = {
    "PHANTOM": {
        "name": "PHANTOM - Studio Ready",
        "type": "fused",
        "models": ["MDX23C", "Kim_Vocal_2"],
        "builtin": True
    },
    "MDX23C": {
        "name": "MDX23C — Highest accuracy",
        "filename": "MDX23C-8KFFT-InstVoc_HQ.ckpt",
        "type": "base",
        "builtin": True
    },
    "Kim_Vocal_2": {
        "name": "Kim Vocal 2 — Clean isolation",
        "filename": "Kim_Vocal_2.onnx",
        "type": "base",
        "builtin": True
    },
    "UVR-MDX-NET-Voc_FT": {
        "name": "UVR-MDX-NET Vocal FT — Balanced",
        "filename": "UVR-MDX-NET-Voc_FT.onnx",
        "type": "base",
        "builtin": True
    }
}


def _load_models_from_folder() -> Dict[str, Any]:
    discovered = {}
    model_dir = _models_dir()
    for ext in (".onnx", ".pth", ".ckpt"):
        for f in model_dir.glob(f"*{ext}"):
            key = f.stem
            discovered[key] = {
                "name": f.stem.replace("_", " ").title(),
                "file": str(f.absolute()),
                "type": "base",
                "builtin": False
            }
    return discovered


def _load_models_json() -> Dict[str, Any]:
    models = BUILTIN_MODELS.copy()
    discovered = _load_models_from_folder()
    models.update(discovered)
    if MODELS_JSON.exists():
        try:
            with open(MODELS_JSON, "r", encoding="utf-8") as f:
                fused = json.load(f)
            for entry in fused:
                if entry.get("type") == "fused" and not entry.get("builtin", False):
                    a_ok = entry["model_a"] in models
                    b_ok = entry["model_b"] in models
                    if a_ok and b_ok:
                        models[entry["key"]] = {
                            "name": entry["name"],
                            "type": "fused",
                            "model_a": entry["model_a"],
                            "model_b": entry["model_b"],
                            "builtin": False
                        }
        except Exception as e:
            print(f"[models.json] Error loading: {e}")
    return models


def _save_fused_model(key: str, name: str, model_a: str, model_b: str):
    fused_list = []
    if MODELS_JSON.exists():
        try:
            with open(MODELS_JSON, "r", encoding="utf-8") as f:
                fused_list = json.load(f)
        except Exception:
            fused_list = []
    fused_list = [e for e in fused_list if e.get("key") != key]
    fused_list.append({
        "key": key,
        "name": name,
        "type": "fused",
        "model_a": model_a,
        "model_b": model_b,
        "builtin": False
    })
    with open(MODELS_JSON, "w", encoding="utf-8") as f:
        json.dump(fused_list, f, indent=2)


def _delete_model_file(key: str):
    models = _load_models_json()
    if key not in models or models[key].get("builtin", False):
        return False
    info = models[key]
    if info["type"] == "base":
        fpath = Path(info["file"])
        if fpath.exists():
            fpath.unlink()
    if info["type"] == "fused":
        fused_list = []
        if MODELS_JSON.exists():
            with open(MODELS_JSON, "r", encoding="utf-8") as f:
                fused_list = json.load(f)
        fused_list = [e for e in fused_list if e.get("key") != key]
        with open(MODELS_JSON, "w", encoding="utf-8") as f:
            json.dump(fused_list, f, indent=2)
    return True


def _is_vocal_model(key: str, info: dict) -> bool:
    """Return True if the model is suitable for vocal extraction."""
    if info.get("type") == "fused" or key == "PHANTOM":
        return True
    name = info.get("name", "").lower()
    key_lower = key.lower()
    inst_keywords = ["inst", "instrumental", "karaoke", "backing"]
    if any(kw in name for kw in inst_keywords) or any(kw in key_lower for kw in inst_keywords):
        return False
    return True


MDX_MODELS_LOOKUP = _load_models_json()
MDX_MODELS = {k: v for k, v in BUILTIN_MODELS.items(
) if v["type"] == "base" or k == "PHANTOM"}

BG = "#0a0a0a"
BG2 = "#111111"
BG3 = "#161616"
BORDER = "#1e1e1e"
BORDER2 = "#252525"
CYAN = "#00e5ff"
CYAN_DIM = "#00b8cc"
CYAN_GLOW = "#00e5ff33"
CYAN_DARK = "#003d4d"
CYAN_MID = "#00c8e0"
TEXT = "#e8f4f8"
TEXT_DIM = "#6a8a90"
TEXT_MUTED = "#3a5a60"
ERROR_C = "#ff4444"
SUCCESS = "#00ff88"
MONO = "'Consolas', 'JetBrains Mono', 'Courier New', monospace"

STYLE = f"""
* {{ font-family: {MONO}; }}
QMainWindow, QWidget {{ background-color: {BG}; color: {TEXT}; }}
QScrollArea {{ border: none; background: transparent; }}
QScrollBar:vertical {{ background: {BG2}; width: 4px; border-radius: 2px; }}
QScrollBar::handle:vertical {{ background: {CYAN_DIM}; border-radius: 2px; min-height: 20px; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar:horizontal {{ height: 0; }}
QTextEdit {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 6px;
    color: {TEXT_DIM}; font-size: 12px; padding: 8px;
    selection-background-color: {CYAN_DARK};
}}
QComboBox {{
    background: {BG2}; border: 1px solid {BORDER2}; border-radius: 6px;
    color: {TEXT}; font-size: 13px; padding: 7px 36px 7px 12px; min-height: 34px;
}}
QComboBox:hover {{ border-color: {CYAN_DIM}; }}
QComboBox:focus {{ border-color: {CYAN}; }}
QComboBox::drop-down {{ border: none; width: 28px; subcontrol-position: right center; }}
QComboBox::down-arrow {{
    width: 0; height: 0; margin-right: 10px;
    border-left: 4px solid transparent; border-right: 4px solid transparent;
    border-top: 5px solid {CYAN_DIM};
}}
QComboBox QAbstractItemView {{
    background: {BG3}; border: 1px solid {BORDER2}; color: {TEXT};
    selection-background-color: {CYAN_DARK}; outline: none; padding: 4px;
}}
QLineEdit {{
    background: {BG2}; border: 1px solid {BORDER2}; border-radius: 6px;
    color: {TEXT}; font-size: 13px; padding: 7px 12px; min-height: 34px;
}}
QLineEdit:hover {{ border-color: {CYAN_DIM}; }}
QLineEdit:focus {{ border-color: {CYAN}; }}
QLineEdit:disabled {{ background: {BG3}; color: {TEXT_MUTED}; border-color: {BORDER}; }}
QCheckBox {{ color: {TEXT}; font-size: 13px; spacing: 8px; }}
QCheckBox::indicator {{
    width: 17px; height: 17px; border: 1px solid {BORDER2};
    border-radius: 4px; background: {BG2};
}}
QCheckBox::indicator:checked {{ background: {CYAN}; border-color: {CYAN}; }}
QCheckBox::indicator:hover {{ border-color: {CYAN_DIM}; }}
QProgressBar {{
    background: {BG2}; border: 1px solid {BORDER}; border-radius: 4px; text-align: center;
}}
QProgressBar::chunk {{
    background: qlineargradient(x1:0,y1:0,x2:1,y2:0, stop:0 {CYAN_DARK}, stop:1 {CYAN});
    border-radius: 3px;
}}
QRadioButton {{ color: {TEXT}; font-size: 13px; spacing: 8px; }}
QRadioButton::indicator {{
    width: 15px; height: 15px; border: 1px solid {BORDER2};
    border-radius: 8px; background: {BG2};
}}
QRadioButton::indicator:checked {{ background: {CYAN}; border-color: {CYAN}; }}
QRadioButton::indicator:hover {{ border-color: {CYAN_DIM}; }}
QLabel {{ background: transparent; }}
QDialog {{ background: {BG}; }}
"""

try:
    import torch
    import numpy as np
    TORCH_OK = True
except ImportError:
    TORCH_OK = False


class HardwareBackend(Enum):
    CUDA = "cuda"
    ROCM = "rocm"
    MPS = "mps"
    OPENVINO = "openvino"
    TENSORRT = "tensorrt"
    CPU = "cpu"


DEREVERB_MODELS = {
    "none": {
        "display": "Disabled",
        "chain":   [],
    },
    "UVR-DeNoise": {
        "display": "De-Noise",
        "chain":   ["UVR-DeNoise.pth"],
    },
    "UVR-DeEcho-DeReverb": {
        "display": "De-Echo / De-Reverb",
        "chain":   ["UVR-DeEcho-DeReverb.pth"],
    },
    "both": {
        "display": "De-Noise → De-Echo / De-Reverb  (Both)",
        "chain":   ["UVR-DeNoise.pth", "UVR-DeEcho-DeReverb.pth"],
    },
    "reverb_HQ": {
        "display": "Reverb HQ",
        "chain":   ["Reverb_HQ_By_FoxJoy.onnx"],
    },
    "triple": {
        "display": "De-Noise → De-Echo/De-Reverb → Reverb HQ  (Triple)",
        "chain":   ["UVR-DeNoise.pth", "UVR-DeEcho-DeReverb.pth", "Reverb_HQ_By_FoxJoy.onnx"],
    },
}

SONGS_JSON = _app_data_dir() / "songs.json"
AUDIO_EXTS = {".wav", ".mp3", ".flac", ".ogg", ".aiff", ".m4a"}
AUDIO_FILTER = "Audio (*.wav *.mp3 *.flac *.ogg *.aiff *.m4a);;All (*)"


def _songs_json_load() -> list:
    if not SONGS_JSON.exists():
        return []
    try:
        with open(SONGS_JSON, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _songs_json_save(records: list):
    try:
        with open(SONGS_JSON, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f"[songs.json] write failed: {e}")


def _songs_json_append(song_name: str, out_dir: Path, stems: dict,
                       preset: str, model: str, backend_desc: str):
    records = _songs_json_load()
    out_str = str(out_dir.resolve())
    records = [r for r in records if r.get("out_dir") != out_str]
    stem_files = {
        name: str(Path(path).resolve())
        for name, path in stems.items()
        if path and Path(path).exists()
    }
    record = {
        "song_name":    song_name,
        "out_dir":      out_str,
        "stems":        stem_files,
        "preset":       preset,
        "model":        model,
        "backend":      backend_desc,
        "date":         datetime.datetime.now().isoformat(timespec="seconds"),
    }
    records.append(record)
    _songs_json_save(records)


def _songs_json_validate_and_prune() -> list:
    records = _songs_json_load()
    valid = []
    for r in records:
        out_dir = Path(r.get("out_dir", ""))
        if not out_dir.is_dir():
            continue
        stems = r.get("stems", {})
        if not stems:
            continue
        missing = [p for p in stems.values() if not Path(p).exists()]
        if missing:
            continue
        valid.append(r)
    if len(valid) != len(records):
        _songs_json_save(valid)
    return valid


CONFIG_TOML = Path(__file__).parent / "config.toml"

_TOML_PRESET_MAP = {
    "fast":        "fast",
    "fine-tuned":  "fine-tuned",
    "fine_tuned":  "fine-tuned",
    "finetuned":   "fine-tuned",
    "pro":         "professional",
    "professional": "professional",
    "max":         "max",
}

_TOML_PP_MAP = {
    "disabled":            "none",
    "none":                "none",
    "denoise":             "UVR-DeNoise",
    "de-noise":            "UVR-DeNoise",
    "de_noise":            "UVR-DeNoise",
    "deecho":              "UVR-DeEcho-DeReverb",
    "de-echo":             "UVR-DeEcho-DeReverb",
    "de-echo-de-reverb":   "UVR-DeEcho-DeReverb",
    "dereverb":            "UVR-DeEcho-DeReverb",
    "de-reverb":           "UVR-DeEcho-DeReverb",
    "both":                "both",
    "reverb_hq":           "reverb_HQ",
    "triple":              "triple",
}


@dataclass
class TomlOverrides:
    shifts:        Optional[int] = None
    overlap:       Optional[float] = None
    vocal_model:   Optional[str] = None
    post_process:  Optional[str] = None
    debleed:       Optional[bool] = None
    export_format: Optional[str] = None


def _load_toml_overrides(preset: str) -> Optional[TomlOverrides]:
    if not _TOML_OK or not CONFIG_TOML.exists():
        return None
    try:
        with open(CONFIG_TOML, "rb") as f:
            data = _toml_lib.load(f)
    except Exception as e:
        print(f"[config.toml] parse error: {e}")
        return None
    section = None
    for raw_key, raw_val in data.items():
        mapped = _TOML_PRESET_MAP.get(raw_key.lower().strip())
        if mapped == preset and isinstance(raw_val, dict):
            section = raw_val
            break
    if section is None:
        return None
    ov = TomlOverrides()
    if "shifts" in section:
        try:
            ov.shifts = max(1, int(section["shifts"]))
        except (ValueError, TypeError):
            pass
    if "overlap" in section:
        try:
            ov.overlap = float(section["overlap"])
        except (ValueError, TypeError):
            pass
    if preset == "max":
        if "vocal_model" in section:
            raw = str(section["vocal_model"]).lower().strip()
            models = _load_models_json()
            for key, info in models.items():
                if info["name"].lower() == raw or key.lower() == raw:
                    ov.vocal_model = key
                    break
        if "post_process" in section:
            raw = str(section["post_process"]
                      ).lower().strip().replace(" ", "-")
            ov.post_process = _TOML_PP_MAP.get(raw)
        if "debleed" in section:
            ov.debleed = bool(section["debleed"])
        if "export_format" in section:
            fmt = str(section["export_format"]).lower().strip()
            if fmt in ("wav", "mp3", "flac"):
                ov.export_format = fmt
    return ov


def _example_config_toml_text() -> str:
    return """\
[fast]
shifts  = 1
overlap = 0.25

[fine-tuned]
shifts  = 2
overlap = 0.25

[pro]
shifts  = 4
overlap = 0.5

[max]
vocal_model   = "PHANTOM"
post_process  = "both"
debleed       = false
export_format = "wav"
"""


def _get_vram_free_mb() -> float:
    if not TORCH_OK:
        return 0.0
    try:
        if torch.cuda.is_available():
            free, _ = torch.cuda.mem_get_info(torch.cuda.current_device())
            return free / 1_048_576
    except Exception:
        pass
    return 0.0


def _get_ram_free_mb() -> float:
    try:
        import psutil
        return psutil.virtual_memory().available / 1_048_576
    except Exception:
        pass
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) / 1024
    except Exception:
        pass
    return 4096.0


def _vram_pressure_high(threshold_mb: float = 1024.0) -> bool:
    free = _get_vram_free_mb()
    return free > 0 and free < threshold_mb


class HardwareDetector:
    @staticmethod
    def detect_cuda() -> bool:
        try:
            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def detect_rocm() -> bool:
        try:
            if hasattr(torch.version, "hip") and torch.version.hip:
                return True
            if torch.cuda.is_available():
                n = torch.cuda.get_device_name(0).lower()
                return "amd" in n or "radeon" in n
        except Exception:
            pass
        return False

    @staticmethod
    def detect_mps() -> bool:
        try:
            return torch.backends.mps.is_available() and torch.backends.mps.is_built()
        except Exception:
            return False

    @staticmethod
    def detect_openvino() -> bool:
        try:
            import openvino
            return True
        except ImportError:
            pass
        try:
            import onnxruntime as ort
            return "OpenVINOExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    @staticmethod
    def detect_tensorrt() -> bool:
        try:
            import onnxruntime as ort
            return "TensorrtExecutionProvider" in ort.get_available_providers()
        except Exception:
            return False

    @staticmethod
    def detect_onnxruntime_gpu() -> bool:
        try:
            import onnxruntime as ort
            gpu = {"CUDAExecutionProvider", "ROCMExecutionProvider",
                   "OpenVINOExecutionProvider", "TensorrtExecutionProvider"}
            return bool(gpu & set(ort.get_available_providers()))
        except Exception:
            return False

    @staticmethod
    def get_best_backend() -> tuple:
        if not TORCH_OK:
            return HardwareBackend.CPU, "CPU (torch not installed)"
        if HardwareDetector.detect_cuda():
            if HardwareDetector.detect_tensorrt():
                try:
                    return HardwareBackend.TENSORRT, f"NVIDIA TensorRT  ({torch.cuda.get_device_name(0)})"
                except Exception:
                    return HardwareBackend.TENSORRT, "NVIDIA TensorRT"
            try:
                return HardwareBackend.CUDA, f"NVIDIA CUDA  ({torch.cuda.get_device_name(0)})"
            except Exception:
                return HardwareBackend.CUDA, "NVIDIA CUDA"
        if HardwareDetector.detect_rocm():
            try:
                return HardwareBackend.ROCM, f"AMD ROCm  ({torch.cuda.get_device_name(0)})"
            except Exception:
                return HardwareBackend.ROCM, "AMD ROCm"
        if HardwareDetector.detect_mps():
            try:
                import platform as _pl
                return HardwareBackend.MPS, f"Apple MPS  ({_pl.processor() or 'Apple Silicon'})"
            except Exception:
                return HardwareBackend.MPS, "Apple MPS"
        if HardwareDetector.detect_openvino():
            try:
                import cpuinfo
                return HardwareBackend.OPENVINO, f"Intel OpenVINO  ({cpuinfo.get_cpu_info().get('brand_raw', 'Intel CPU')})"
            except Exception:
                return HardwareBackend.OPENVINO, "Intel OpenVINO"
        try:
            import cpuinfo
            return HardwareBackend.CPU, f"CPU  ({cpuinfo.get_cpu_info().get('brand_raw', 'CPU')})"
        except Exception:
            return HardwareBackend.CPU, "CPU"

    @staticmethod
    def get_onnx_providers(backend: HardwareBackend) -> list:
        try:
            import onnxruntime as ort
            av = ort.get_available_providers()
        except Exception:
            return ["CPUExecutionProvider"]
        if backend == HardwareBackend.TENSORRT and "TensorrtExecutionProvider" in av:
            return ["TensorrtExecutionProvider", "CUDAExecutionProvider", "CPUExecutionProvider"]
        if backend in (HardwareBackend.CUDA, HardwareBackend.TENSORRT) and "CUDAExecutionProvider" in av:
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        if backend == HardwareBackend.ROCM and "ROCMExecutionProvider" in av:
            return ["ROCMExecutionProvider", "CPUExecutionProvider"]
        if backend == HardwareBackend.OPENVINO and "OpenVINOExecutionProvider" in av:
            return ["OpenVINOExecutionProvider", "CPUExecutionProvider"]
        return ["CPUExecutionProvider"]


@dataclass
class MaxModeOptions:
    mdx_model_key:      str = "PHANTOM"
    dereverb_model_key: Optional[str] = "both"
    debleed:            bool = False


@dataclass
class SeparationConfig:
    quality_preset: Literal["fast", "fine-tuned", "professional", "max"]
    model:          str
    shifts:         int
    overlap:        float
    use_mdx_vocals: bool
    device:         str
    backend:        HardwareBackend
    onnx_providers: list
    export_format:  str = "wav"
    max_options:    Optional[MaxModeOptions] = None
    seven_stem:     bool = False

    @classmethod
    def from_preset(cls, preset: str, backend=None, max_options=None, export_format="wav",
                    toml_overrides: Optional["TomlOverrides"] = None,
                    seven_stem: bool = False):
        if backend is None:
            backend, _ = HardwareDetector.get_best_backend()
        device = (
            "cuda" if backend in (HardwareBackend.CUDA, HardwareBackend.ROCM, HardwareBackend.TENSORRT)
            else "mps" if backend == HardwareBackend.MPS
            else "cpu"
        )
        onnx = HardwareDetector.get_onnx_providers(backend)
        gpu = backend in (HardwareBackend.CUDA,
                          HardwareBackend.ROCM, HardwareBackend.TENSORRT)
        gs, cs = _resource_aware_shifts(preset, gpu)
        if seven_stem:
            base_model = "htdemucs_6s"
        else:
            base_model = "htdemucs_ft" if preset != "fast" else "htdemucs"
        table = {
            "fast":         dict(model=base_model if seven_stem else "htdemucs",
                                 shifts=1, overlap=0.25, use_mdx_vocals=False),
            "fine-tuned":   dict(model=base_model if seven_stem else "htdemucs_ft",
                                 shifts=cs, overlap=0.25, use_mdx_vocals=False),
            "professional": dict(model=base_model if seven_stem else "htdemucs_ft",
                                 shifts=gs if gpu else cs, overlap=0.5, use_mdx_vocals=not seven_stem),
            "max":          dict(model=base_model if seven_stem else "htdemucs_ft",
                                 shifts=gs if gpu else cs, overlap=0.5, use_mdx_vocals=not seven_stem),
        }
        kw = table.get(preset, table["fine-tuned"])
        if toml_overrides is not None:
            if toml_overrides.shifts is not None:
                kw["shifts"] = toml_overrides.shifts
            if toml_overrides.overlap is not None:
                kw["overlap"] = toml_overrides.overlap
            if toml_overrides.export_format is not None:
                export_format = toml_overrides.export_format
            if preset == "max" and not seven_stem and toml_overrides.vocal_model is not None:
                _mo = max_options or MaxModeOptions()
                if toml_overrides.vocal_model:
                    _mo.mdx_model_key = toml_overrides.vocal_model
                if toml_overrides.post_process is not None:
                    _mo.dereverb_model_key = None if toml_overrides.post_process == "none" else toml_overrides.post_process
                if toml_overrides.debleed is not None:
                    _mo.debleed = toml_overrides.debleed
                max_options = _mo
        cfg = cls(quality_preset=preset, device=device, backend=backend, onnx_providers=onnx,
                  max_options=max_options or MaxModeOptions(), export_format=export_format,
                  seven_stem=seven_stem, **kw)
        if backend == HardwareBackend.CPU and not HardwareDetector.detect_onnxruntime_gpu():
            cfg.use_mdx_vocals = False
        return cfg


def _resource_aware_shifts(preset: str, gpu: bool) -> tuple:
    vram_free = _get_vram_free_mb()
    if vram_free > 0:
        gpu_shifts = 4 if vram_free >= 8192 else 3 if vram_free >= 4096 else 2 if vram_free >= 2048 else 1
    else:
        gpu_shifts = 4
    ram_free = _get_ram_free_mb()
    cpu_shifts = 2 if ram_free >= 8192 else 1
    if preset in ("fast", "fine-tuned"):
        gpu_shifts = min(gpu_shifts, 2)
    return gpu_shifts, cpu_shifts


def _phase_aware_merge(stems: list, samplerate: int) -> "np.ndarray":
    try:
        import numpy as _np
    except ImportError:
        result = None
        for s in stems:
            result = s if result is None else result + s
        return result
    if not stems:
        return None
    arrays = []
    for s in stems:
        if s.ndim == 1:
            s = s[:, None]
        arrays.append(s)
    min_len = min(a.shape[0] for a in arrays)
    arrays = [a[:min_len] for a in arrays]
    n_ch = max(a.shape[1] for a in arrays)
    win_size = 2048
    hop = win_size // 4
    window = np.sqrt(np.hanning(win_size)).astype(np.float64)
    n_samp = min_len
    combined_specs = None
    for arr in arrays:
        if arr.shape[1] < n_ch:
            arr = np.repeat(arr, n_ch, axis=1)
        specs = []
        for ch in range(n_ch):
            sig = arr[:, ch].astype(np.float64)
            spec = np.array([
                np.fft.rfft(sig[pos:pos + win_size] * window)
                for pos in range(0, n_samp - win_size + 1, hop)
            ])
            specs.append(spec)
        specs = np.stack(specs, axis=0)
        combined_specs = specs if combined_specs is None else combined_specs + specs
    if combined_specs is None:
        return np.zeros((min_len, n_ch), dtype=np.float32)
    out = np.zeros((n_samp, n_ch), dtype=np.float64)
    for ch in range(n_ch):
        out_ch = np.zeros(n_samp, dtype=np.float64)
        acc_ch = np.zeros(n_samp, dtype=np.float64)
        for fi, pos in enumerate(range(0, n_samp - win_size + 1, hop)):
            frame = np.fft.irfft(combined_specs[ch, fi], n=win_size)
            out_ch[pos:pos + win_size] += frame * window
            acc_ch[pos:pos + win_size] += window ** 2
        out[:, ch] = out_ch / np.where(acc_ch > 1e-8, acc_ch, 1.0)
    peak = np.max(np.abs(out))
    if peak > 0.98:
        out = out / peak * 0.98
    return out.astype(np.float32)


def _phantom_spectral_blend(arrays: list, samplerate: int) -> "np.ndarray":
    import numpy as _np
    if len(arrays) == 1:
        a = arrays[0]
        if a.ndim == 1:
            a = a[:, None]
        return a.astype(_np.float32)
    processed = []
    for a in arrays:
        if a.ndim == 1:
            a = a[:, None]
        processed.append(a)
    min_len = min(a.shape[0] for a in processed)
    processed = [a[:min_len].astype(_np.float64) for a in processed]
    n_ch = max(a.shape[1] for a in processed)
    padded = []
    for a in processed:
        if a.shape[1] < n_ch:
            a = _np.repeat(a, n_ch, axis=1)
        padded.append(a)
    win_size = 4096
    hop = win_size // 8
    window = _np.hanning(win_size).astype(_np.float64)
    n_samp = min_len
    n_bins = win_size // 2 + 1
    freqs = _np.fft.rfftfreq(win_size, d=1.0 / max(samplerate, 44100))
    lo, hi = 80.0, 8000.0
    band_w = _np.ones(n_bins, dtype=_np.float64)
    with _np.errstate(divide="ignore", invalid="ignore"):
        below = freqs < lo
        above = freqs > hi
        band_w[below] = _np.where(
            freqs[below] > 0,
            _np.clip((freqs[below] / lo) ** 0.4, 0.05, 1.0),
            0.05,
        )
        band_w[above] = _np.clip((hi / freqs[above]) ** 0.3, 0.05, 1.0)
    band_w = band_w[_np.newaxis, :]
    eps = 1e-9
    pad = win_size - ((n_samp - win_size) % hop or hop)
    padded_len = n_samp + pad
    frames = list(range(0, padded_len - win_size + 1, hop))
    out = _np.zeros((n_samp, n_ch), dtype=_np.float64)
    for ch in range(n_ch):
        specs = []
        for arr in padded:
            sig = _np.pad(arr[:, ch], (0, pad))
            S = _np.stack([
                _np.fft.rfft(sig[pos:pos + win_size] * window)
                for pos in frames
            ])
            specs.append(S)
        biased_mags = [_np.abs(S) * band_w for S in specs]
        total_mag = sum(biased_mags) + eps
        blended = sum((bm / total_mag) * S for bm,
                      S in zip(biased_mags, specs))
        out_ch = _np.zeros(padded_len, dtype=_np.float64)
        acc_ch = _np.zeros(padded_len, dtype=_np.float64)
        for fi, pos in enumerate(frames):
            frame = _np.fft.irfft(blended[fi], n=win_size)
            out_ch[pos:pos + win_size] += frame * window
            acc_ch[pos:pos + win_size] += window ** 2
        out[:, ch] = (out_ch / _np.where(acc_ch > 1e-8, acc_ch, 1.0))[:n_samp]
    for ch in range(n_ch):
        peak = _np.max(_np.abs(out[:, ch]))
        if peak > 1e-6:
            out[:, ch] = out[:, ch] / peak * 0.944
    return out.astype(_np.float32)


def _phantom_blend_files(paths: list, output_path: Path, samplerate_hint: int = 44100) -> Path:
    import soundfile as sf
    import numpy as _np
    arrays, sr_ref = [], None
    for p in paths:
        a, sr = sf.read(str(p))
        if sr_ref is None:
            sr_ref = sr
        if a.ndim == 1:
            a = a[:, None]
        arrays.append(a)
    merged = _phase_aware_merge(arrays, sr_ref)
    if merged is None:
        merged = _np.zeros((sr_ref * 2, 2), dtype=_np.float32)
    sf.write(str(output_path), merged, sr_ref, subtype="PCM_24")
    return output_path


def load_waveform_data(file_path: str, num_bars: int = 300) -> list:
    try:
        import soundfile as sf
        import numpy as _np
        data, sr = sf.read(str(file_path))
        if data.ndim > 1:
            data = data.mean(axis=1)
        chunk = max(1, len(data) // num_bars)
        bars = []
        for i in range(num_bars):
            start = i * chunk
            end = start + chunk
            seg = data[start:end]
            if len(seg) == 0:
                bars.append(0.0)
            else:
                rms = float(_np.sqrt(_np.mean(seg ** 2)))
                bars.append(rms)
        peak = max(bars) if bars else 1.0
        if peak > 0:
            bars = [b / peak for b in bars]
        return bars
    except Exception:
        return [0.3] * num_bars


_STEMS_6S = ["drums", "bass", "other", "vocals", "guitar", "piano"]


class STEMSeparatorLogic:
    _VR_EXT = {".pth"}
    _MDX_EXT = {".onnx"}
    _MDXC_EXT = {".ckpt"}

    def __init__(self, config: SeparationConfig, progress_cb=None):
        self.config = config
        self.progress_cb = progress_cb or (lambda _: None)
        self.demucs_model = None
        self.models_lookup = _load_models_json()

    def _free_gpu(self, force: bool = False):
        is_gpu = self.config.backend in (
            HardwareBackend.CUDA, HardwareBackend.ROCM, HardwareBackend.TENSORRT)
        if not is_gpu:
            gc.collect()
            return
        if force or _vram_pressure_high(threshold_mb=1024.0):
            gc.collect()
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
        elif self.config.backend == HardwareBackend.MPS:
            try:
                torch.mps.empty_cache()
            except Exception:
                pass

    def _maybe_free_gpu(self):
        self._free_gpu(force=False)

    def _apply_tensorrt_optimisation(self, model_filename: str) -> str:
        if self.config.backend != HardwareBackend.TENSORRT:
            return model_filename
        if Path(model_filename).suffix.lower() not in self._MDX_EXT:
            return model_filename
        try:
            import onnxruntime as ort
            cache_dir = Path("/tmp/trt_engine_cache")
            cache_dir.mkdir(parents=True, exist_ok=True)
            trt_opts = ort.SessionOptions()
            provider_opts = {
                "trt_engine_cache_enable": True,
                "trt_engine_cache_path":   str(cache_dir),
                "trt_fp16_enable":         True,
            }
            ort.InferenceSession(
                model_filename, sess_options=trt_opts,
                providers=[("TensorrtExecutionProvider", provider_opts),
                           "CUDAExecutionProvider", "CPUExecutionProvider"],
            )
        except Exception as e:
            print(f"  [trt] Optimisation skipped: {e}")
        return model_filename

    def _load_demucs(self):
        if self.demucs_model:
            return
        from demucs.pretrained import get_model
        print(f"▸ Loading Demucs model [{self.config.model}]")
        self.demucs_model = get_model(self.config.model)
        self.demucs_model.to(self.config.device)
        self.demucs_model.eval()
        if self.config.backend in (HardwareBackend.CUDA, HardwareBackend.TENSORRT):
            torch.backends.cudnn.benchmark = True
        elif self.config.backend == HardwareBackend.OPENVINO:
            torch.set_num_threads(os.cpu_count() or 4)
        print(f"✓ Demucs ready on {self.config.device.upper()}")

    def _stem_ext(self):
        return self.config.export_format

    def _wav_to_mp3(self, src: Path, dst: Path) -> bool:
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i", str(src), "-b:a", "320k", str(dst)],
                capture_output=True)
            if r.returncode == 0:
                return True
        except FileNotFoundError:
            pass
        try:
            from pydub import AudioSegment
            AudioSegment.from_wav(str(src)).export(
                str(dst), format="mp3", bitrate="320k")
            return True
        except Exception:
            pass
        return False

    def _wav_to_flac(self, src: Path, dst: Path) -> bool:
        try:
            r = subprocess.run(
                ["ffmpeg", "-y", "-i",
                    str(src), "-c:a", "flac", "-compression_level", "8", str(dst)],
                capture_output=True)
            if r.returncode == 0:
                return True
        except FileNotFoundError:
            pass
        try:
            from pydub import AudioSegment
            AudioSegment.from_wav(str(src)).export(str(dst), format="flac")
            return True
        except Exception:
            pass
        try:
            import soundfile as sf
            data, sr = sf.read(str(src))
            sf.write(str(dst), data, sr, format='flac', subtype='PCM_24')
            return True
        except Exception:
            return False

    def _commit_vocals(self, wav_path: Path, output_dir: Path) -> Path:
        ext = self._stem_ext()
        target = output_dir / f"vocals.{ext}"
        if ext == "mp3":
            ok = self._wav_to_mp3(wav_path, target)
            wav_path.unlink(missing_ok=True)
            if not ok:
                target = output_dir / "vocals.wav"
                if wav_path.resolve() != target.resolve():
                    shutil.move(str(wav_path), str(target))
        elif ext == "flac":
            ok = self._wav_to_flac(wav_path, target)
            wav_path.unlink(missing_ok=True)
            if not ok:
                target = output_dir / "vocals.wav"
                if wav_path.resolve() != target.resolve():
                    shutil.move(str(wav_path), str(target))
        else:
            if wav_path.resolve() != target.resolve():
                shutil.move(str(wav_path), str(target))
        print(f"✓ Saved {target.name}")
        return target

    def _save_audio(self, src, name: str, out_dir: Path, sr: int) -> Path:
        from demucs.audio import save_audio
        wav_p = out_dir / f"{name}.wav"
        save_audio(src, str(wav_p), samplerate=sr, bitrate=320,
                   clip="rescale", as_float=False, bits_per_sample=24)
        ext = self._stem_ext()
        if ext == "mp3":
            mp3_p = out_dir / f"{name}.mp3"
            ok = self._wav_to_mp3(wav_p, mp3_p)
            wav_p.unlink(missing_ok=True)
            if ok:
                print(f"✓ Saved {mp3_p.name}")
                return mp3_p
            print("⚠ MP3 encode failed — saved as WAV")
            return wav_p
        elif ext == "flac":
            flac_p = out_dir / f"{name}.flac"
            ok = self._wav_to_flac(wav_p, flac_p)
            wav_p.unlink(missing_ok=True)
            if ok:
                print(f"✓ Saved {flac_p.name}")
                return flac_p
            print("⚠ FLAC encode failed — saved as WAV")
            return wav_p
        else:
            print(f"✓ Saved {wav_p.name}")
            return wav_p

    def _set_ort_env(self):
        m = {
            HardwareBackend.CUDA:     "CUDAExecutionProvider",
            HardwareBackend.TENSORRT: "TensorrtExecutionProvider",
            HardwareBackend.ROCM:     "ROCMExecutionProvider",
            HardwareBackend.OPENVINO: "OpenVINOExecutionProvider",
        }
        ep = m.get(self.config.backend)
        if ep:
            os.environ["ORT_EXECUTION_PROVIDER"] = ep

    def _make_separator(self, model_filename: str):
        from audio_separator.separator import Separator
        self._set_ort_env()
        self._maybe_free_gpu()
        model_filename = self._apply_tensorrt_optimisation(model_filename)
        model_dir = str(_models_dir())
        print(f"  Loading model: {model_filename} from {model_dir}")
        ext = Path(model_filename).suffix.lower()
        kw = dict(model_file_dir=model_dir, output_format="wav",
                  normalization_threshold=0.9, amplification_threshold=0.9)
        if ext in self._VR_EXT:
            kw["vr_params"] = dict(batch_size=8, window_size=512, aggression=5,
                                   enable_tta=False, enable_post_process=False,
                                   post_process_threshold=0.2, high_end_process=False)
        elif ext in self._MDXC_EXT:
            kw["mdxc_params"] = dict(
                segment_size=256, overlap=8, batch_size=1, pitch_shift=0)
        else:
            kw["mdx_params"] = dict(
                hop_length=1024, segment_size=256, overlap=0.25, batch_size=1)
        sep = Separator(**kw)
        base_name = Path(model_filename).name
        sep.load_model(model_filename=base_name)
        return sep

    def _run_separator_get_vocal(self, audio_path: Path, model_filename: str, dest: Path) -> Path:
        sep = self._make_separator(model_filename)
        files = sep.separate(str(audio_path))
        vf = next((Path(f)
                  for f in files if "Vocals" in f or "vocals" in f), None)
        if vf is None:
            raise RuntimeError(
                f"Model {model_filename} produced no vocal stem")
        if vf.resolve() != dest.resolve():
            shutil.move(str(vf), str(dest))
        for f in files:
            pp = Path(f)
            if pp.exists() and pp.resolve() != dest.resolve():
                pp.unlink(missing_ok=True)
        return dest

    def separate_with_demucs(self, audio_path: Path, output_dir: Path) -> dict:
        from demucs.audio import AudioFile
        from demucs.apply import apply_model
        self._load_demucs()
        print(
            f"▸ Demucs separation (shifts={self.config.shifts}, overlap={self.config.overlap})")
        wav = AudioFile(str(audio_path)).read(
            streams=0, samplerate=self.demucs_model.samplerate,
            channels=self.demucs_model.audio_channels)
        ref = wav.mean(0)
        wav = (wav - ref.mean()) / wav.std()
        sources = apply_model(
            self.demucs_model, wav[None], device=self.config.device,
            shifts=self.config.shifts, overlap=self.config.overlap, progress=True)[0]
        sources = sources * wav.std() + ref.mean()
        output_dir.mkdir(parents=True, exist_ok=True)
        self._maybe_free_gpu()
        sr = self.demucs_model.samplerate
        stem_names = _STEMS_6S if self.config.seven_stem else [
            "drums", "bass", "other", "vocals"]
        return {name: self._save_audio(sources[i], name, output_dir, sr)
                for i, name in enumerate(stem_names)}

    def separate_vocals_mdx(self, audio_path: Path, output_dir: Path, model_filename: str) -> Path:
        print(f"▸ MDX vocal extraction [{model_filename}]")
        tmp = output_dir / "_vox_work.wav"
        self._run_separator_get_vocal(audio_path, model_filename, tmp)
        return tmp

    def separate_vocals_phantom(self, audio_path: Path, output_dir: Path) -> Path:
        print("▸ PHANTOM — dual-model spectral fusion [MDX23C × Kim Vocal 2]")
        tmp_dir = output_dir / "_phantom_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        vocal_paths = []
        for i, key in enumerate(["MDX23C", "Kim_Vocal_2"]):
            fn = BUILTIN_MODELS[key]["filename"]
            dest = tmp_dir / f"phantom_vox_{i}.wav"
            print(f"  [PHANTOM  {i+1}/2] {key}")
            self._run_separator_get_vocal(audio_path, fn, dest)
            vocal_paths.append(dest)
            self._maybe_free_gpu()
        print("  [PHANTOM] Spectral fusion...")
        fused_wav = tmp_dir / "phantom_fused.wav"
        _phantom_blend_files(vocal_paths, fused_wav)
        wav_dest = output_dir / "_vox_work.wav"
        shutil.copy2(str(fused_wav), str(wav_dest))
        try:
            shutil.rmtree(str(tmp_dir))
        except Exception:
            pass
        return wav_dest

    def separate_vocals_fused(self, audio_path: Path, output_dir: Path,
                              model_a_key: str, model_b_key: str) -> Path:
        print(f"▸ Fused model: {model_a_key} × {model_b_key}")
        tmp_dir = output_dir / "_fuse_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        vocal_paths = []
        models = _load_models_json()
        for i, key in enumerate([model_a_key, model_b_key]):
            info = models[key]
            if info["type"] != "base":
                raise ValueError("Fused model can only consist of base models")
            fn = info.get("file", info.get("filename"))
            base_fn = Path(fn).name
            dest = tmp_dir / f"fuse_vox_{i}.wav"
            print(f"  [Fuse {i+1}/2] {key}")
            self._run_separator_get_vocal(audio_path, base_fn, dest)
            vocal_paths.append(dest)
            self._maybe_free_gpu()
        print("  [Fuse] Spectral fusion...")
        fused_wav = tmp_dir / "fused_result.wav"
        _phantom_blend_files(vocal_paths, fused_wav)
        wav_dest = output_dir / "_vox_work.wav"
        shutil.copy2(str(fused_wav), str(wav_dest))
        try:
            shutil.rmtree(str(tmp_dir))
        except Exception:
            pass
        return wav_dest

    def apply_postprocess_chain(self, vocal_path: Path, output_dir: Path, chain: list) -> Path:
        if not chain:
            return vocal_path
        CLEAN_TERMS = [
            "(no noise)", "(no echo)", "(no reverb)", "(vocals)",
            "no noise", "no echo", "no reverb",
        ]
        tmp_dir = output_dir / "_postprocess_tmp"
        tmp_dir.mkdir(parents=True, exist_ok=True)
        current = vocal_path
        for step_idx, model_fn in enumerate(chain):
            print(f"  [Post {step_idx + 1}/{len(chain)}] {model_fn}")
            sep = self._make_separator(model_fn)
            files = sep.separate(str(current))
            if not files:
                continue
            existing = [Path(f) for f in files if Path(f).exists()]
            cleaned = next(
                (Path(f) for term in CLEAN_TERMS
                 for f in files if term in Path(f).name.lower()), None)
            if cleaned is None or not cleaned.exists():
                cleaned = min(existing, key=lambda p: p.stat().st_size)
            step_out = tmp_dir / f"post_step_{step_idx}.wav"
            if cleaned.resolve() != step_out.resolve():
                shutil.move(str(cleaned), str(step_out))
            for pp in existing:
                if pp.exists() and pp.resolve() != step_out.resolve():
                    pp.unlink(missing_ok=True)
            current = step_out
            self._maybe_free_gpu()
        wav_dest = output_dir / "_vox_work.wav"
        if current.resolve() != wav_dest.resolve():
            shutil.copy2(str(current), str(wav_dest))
        try:
            shutil.rmtree(str(tmp_dir))
        except Exception:
            pass
        return wav_dest

    def apply_debleed(self, vocal_path: Path, instrumental_path: Path, output_dir: Path) -> Path:
        import soundfile as sf
        print("▸ De-bleed")
        vocals,  sr_v = sf.read(str(vocal_path))
        instrum, sr_i = sf.read(str(instrumental_path))
        if sr_v != sr_i:
            return vocal_path
        n = min(len(vocals), len(instrum))
        vocals = vocals[:n].copy()
        instrum = instrum[:n].copy()
        if vocals.ndim == 1:
            vocals = vocals[:, np.newaxis]
        if instrum.ndim == 1:
            instrum = instrum[:, np.newaxis]
        if vocals.shape[1] != instrum.shape[1]:
            instrum = np.repeat(
                np.mean(instrum, axis=1, keepdims=True), vocals.shape[1], axis=1)
        ws, hop, alpha = 2048, 512, 0.03
        win = np.sqrt(np.hanning(ws)).astype(np.float64)
        ns = vocals.shape[0]
        result = np.zeros_like(vocals, dtype=np.float64)
        for ch in range(vocals.shape[1]):
            v = vocals[:, ch].astype(np.float64)
            ins = instrum[:, ch].astype(np.float64)
            out = np.zeros(ns, dtype=np.float64)
            nacc = np.zeros(ns, dtype=np.float64)
            for pos in range(0, ns - ws + 1, hop):
                V = np.fft.rfft(v[pos:pos + ws] * win)
                I = np.fft.rfft(ins[pos:pos + ws] * win)
                Vp = np.abs(V) ** 2
                Ip = np.abs(I) ** 2
                mask = Vp / np.maximum(Vp + alpha * Ip, 1e-12)
                out[pos:pos + ws] += np.fft.irfft(V * mask, n=ws) * win
                nacc[pos:pos + ws] += win ** 2
            result[:, ch] = out / np.where(nacc > 1e-8, nacc, 1.0)
        peak = np.max(np.abs(result))
        if peak <= 1e-6:
            return vocal_path
        if peak > 0.98:
            result = result / peak * 0.98
        import soundfile as sf2
        wav_dest = output_dir / "_vox_work.wav"
        sf2.write(str(wav_dest), result.astype(
            np.float32), sr_v, subtype="PCM_24")
        return wav_dest

    def mix_stems(self, paths: list, out: Path, sr: int):
        import soundfile as sf
        arrays = [sf.read(str(p))[0] for p in paths]
        merged = _phase_aware_merge(arrays, sr)
        if merged is None:
            merged = np.zeros((sr * 2, 2), dtype=np.float32)
        ext = self._stem_ext()
        if ext == "mp3":
            tmp = out.with_suffix(".mix_tmp.wav")
            sf.write(str(tmp), merged, sr, subtype="PCM_24")
            ok = self._wav_to_mp3(tmp, out)
            tmp.unlink(missing_ok=True)
            if not ok:
                out = out.with_suffix(".wav")
                sf.write(str(out), merged, sr, subtype="PCM_24")
        elif ext == "flac":
            tmp = out.with_suffix(".mix_tmp.wav")
            sf.write(str(tmp), merged, sr, subtype="PCM_24")
            ok = self._wav_to_flac(tmp, out)
            tmp.unlink(missing_ok=True)
            if not ok:
                out = out.with_suffix(".wav")
                sf.write(str(out), merged, sr, subtype="PCM_24")
        else:
            sf.write(str(out), merged, sr, subtype="PCM_24")
        print(f"✓ Saved {out.name}")

    def separate(self, audio_path: Path, output_dir: Path) -> dict:
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        if not audio_path.exists():
            raise FileNotFoundError(f"Not found: {audio_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_cb(5)
        ext = self._stem_ext()
        if self.config.seven_stem:
            stems = self.separate_with_demucs(audio_path, output_dir)
            self.progress_cb(70)
            inst_sources = [v for k, v in stems.items() if k != "vocals"]
            inst = output_dir / f"instrumental.{ext}"
            self.mix_stems(inst_sources, inst, self.demucs_model.samplerate)
            self.progress_cb(100)
            return {**stems, "instrumental": inst}
        if not self.config.use_mdx_vocals:
            stems = self.separate_with_demucs(audio_path, output_dir)
            self.progress_cb(70)
            inst = output_dir / f"instrumental.{ext}"
            self.mix_stems([stems["bass"], stems["drums"], stems["other"]], inst,
                           self.demucs_model.samplerate)
            self.progress_cb(100)
            return {**stems, "instrumental": inst}
        is_max = self.config.quality_preset == "max"
        opts = self.config.max_options if is_max else None
        model_key = opts.mdx_model_key if is_max else "UVR-MDX-NET-Voc_FT"
        models = _load_models_json()
        model_info = models.get(model_key, BUILTIN_MODELS.get(model_key))

        if model_key == "PHANTOM":
            mdx_vox = self.separate_vocals_phantom(audio_path, output_dir)
        elif model_info and model_info.get("type") == "fused":
            mdx_vox = self.separate_vocals_fused(
                audio_path, output_dir,
                model_info["model_a"], model_info["model_b"])
        else:
            fn = model_info.get("file", model_info.get("filename"))
            base_fn = Path(fn).name
            mdx_vox = self.separate_vocals_mdx(audio_path, output_dir, base_fn)
        self.progress_cb(30)
        stems = self.separate_with_demucs(audio_path, output_dir)
        self.progress_cb(65)
        self._free_gpu(force=True)
        inst = output_dir / f"instrumental.{ext}"
        self.mix_stems([stems["bass"], stems["drums"], stems["other"]], inst,
                       self.demucs_model.samplerate)
        self.progress_cb(75)
        if is_max and opts and opts.dereverb_model_key:
            chain = DEREVERB_MODELS.get(
                opts.dereverb_model_key, {}).get("chain", [])
            if chain:
                mdx_vox = self.apply_postprocess_chain(
                    mdx_vox, output_dir, chain)
        self.progress_cb(90)
        if is_max and opts and opts.debleed:
            mdx_vox = self.apply_debleed(mdx_vox, inst, output_dir)
        mdx_vox = self._commit_vocals(mdx_vox, output_dir)
        self.progress_cb(100)
        return {
            "vocals":       mdx_vox,
            "bass":         stems["bass"],
            "drums":        stems["drums"],
            "other":        stems["other"],
            "instrumental": inst,
        }


def run_gui():
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QLabel, QPushButton, QFileDialog, QComboBox, QCheckBox,
        QStackedWidget, QFrame, QProgressBar, QTextEdit, QScrollArea,
        QLineEdit, QSlider, QMessageBox, QRadioButton, QDialog,
        QGridLayout, QSizePolicy, QListWidget, QListWidgetItem,
        QAbstractItemView, QDialogButtonBox,
    )
    from PyQt6.QtCore import (
        Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve,
        QTimer, pyqtProperty, QUrl, QRect, QRectF, QPointF,
    )
    from PyQt6.QtGui import (
        QFont, QColor, QPalette, QPainter, QPen, QBrush, QCursor,
        QLinearGradient, QRadialGradient,
    )
    from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput

    class WaveformWidget(QWidget):
        seek_requested = pyqtSignal(float)

        def __init__(self, colour: str = CYAN, parent=None):
            super().__init__(parent)
            self.colour = QColor(colour)
            self._bars: list = []
            self._progress: float = 0.0
            self._hover_pos: float = -1.0
            self.setFixedHeight(72)
            self.setMinimumWidth(200)
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.setMouseTracking(True)
            self._blink = 0.0
            self._timer = QTimer(self)
            self._timer.setInterval(33)
            self._timer.timeout.connect(self._tick)
            self._timer.start()

        def set_bars(self, bars: list):
            self._bars = bars
            self.update()

        def set_progress(self, frac: float):
            self._progress = max(0.0, min(1.0, frac))
            self.update()

        def _tick(self):
            self._blink = (self._blink + 0.05) % (2 * math.pi)
            self.update()

        def mouseMoveEvent(self, e):
            self._hover_pos = e.position().x() / max(self.width(), 1)
            self.update()

        def leaveEvent(self, e):
            self._hover_pos = -1.0
            self.update()

        def mousePressEvent(self, e):
            frac = e.position().x() / max(self.width(), 1)
            self.seek_requested.emit(max(0.0, min(1.0, frac)))

        def paintEvent(self, _):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            w = self.width()
            h = self.height()
            bars = self._bars
            n = len(bars)
            if n == 0:
                p.fillRect(0, 0, w, h, QColor(BG2))
                p.setPen(QColor(BORDER2))
                mid = h // 2
                p.drawLine(0, mid, w, mid)
                return
            p.fillRect(0, 0, w, h, QColor(BG2))
            bar_w = max(1.0, w / n)
            gap = max(0.5, bar_w * 0.25)
            bw = max(1.0, bar_w - gap)
            cx = h / 2
            playhead_x = int(self._progress * w)
            accent = QColor(self.colour)
            dim = QColor(self.colour)
            dim.setAlpha(55)
            for i, amp in enumerate(bars):
                x = i * bar_w
                bar_h = max(2.0, amp * (h - 8))
                top = cx - bar_h / 2
                rect = QRectF(x + gap / 2, top, bw, bar_h)
                if x + bw < playhead_x:
                    grad = QLinearGradient(
                        rect.left(), rect.top(), rect.left(), rect.bottom())
                    c1 = QColor(self.colour)
                    c1.setAlpha(220)
                    c2 = QColor(self.colour)
                    c2.setAlpha(120)
                    grad.setColorAt(0.0, c1)
                    grad.setColorAt(1.0, c2)
                    p.fillRect(rect, grad)
                else:
                    p.fillRect(rect, dim)
            if 0.0 <= self._hover_pos <= 1.0:
                hx = int(self._hover_pos * w)
                hc = QColor(self.colour)
                hc.setAlpha(90)
                p.setPen(QPen(hc, 1))
                p.drawLine(hx, 0, hx, h)
            ph_alpha = int(180 + 60 * math.sin(self._blink))
            ph_col = QColor(self.colour)
            ph_col.setAlpha(ph_alpha)
            p.setPen(QPen(ph_col, 2))
            p.drawLine(playhead_x, 0, playhead_x, h)
            p.setPen(QPen(ph_col, 3))
            p.drawLine(playhead_x - 1, 0, playhead_x + 1, 4)
            p.drawLine(playhead_x - 1, h, playhead_x + 1, h - 4)
            p.end()

    class PreviewWorker(QThread):
        log_line = pyqtSignal(str)
        progress = pyqtSignal(int)
        finished = pyqtSignal(bool, str, str)

        CLIP_DURATION = 10

        def __init__(self, config: SeparationConfig, audio_path: Path, parent=None):
            super().__init__(parent)
            self.config = config
            self.audio_path = audio_path
            self._tmp_dir: Optional[Path] = None

        def _trim_audio(self, src: Path, dest: Path, start_s: float, dur_s: float):
            try:
                r = subprocess.run(
                    ["ffmpeg", "-y", "-ss", str(start_s), "-t", str(dur_s),
                     "-i", str(src), "-ar", "44100", "-ac", "2", str(dest)],
                    capture_output=True)
                if r.returncode == 0:
                    return
            except FileNotFoundError:
                pass
            import soundfile as sf
            import numpy as _np
            data, sr = sf.read(str(src))
            s0 = int(start_s * sr)
            s1 = s0 + int(dur_s * sr)
            clip = data[s0:s1]
            if clip.ndim == 1:
                clip = clip[:, _np.newaxis]
            sf.write(str(dest), clip, sr, subtype="PCM_24")

        def run(self):
            old = sys.stdout
            try:
                sig = self.log_line

                class Tee:
                    def __init__(self): self.buf = ""

                    def write(self, t):
                        self.buf += t
                        while "\n" in self.buf:
                            line, self.buf = self.buf.split("\n", 1)
                            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
                            if clean.strip():
                                sig.emit(clean)

                    def flush(self): pass
                sys.stdout = Tee()
                self._tmp_dir = Path(
                    tempfile.mkdtemp(prefix="stemlab_preview_"))
                clip_path = self._tmp_dir / "preview_clip.wav"
                out_dir = self._tmp_dir / "out"
                out_dir.mkdir()
                try:
                    import soundfile as sf
                    info = sf.info(str(self.audio_path))
                    total = info.duration
                except Exception:
                    total = 120.0
                start = max(0.0, total * 0.25)
                if start + self.CLIP_DURATION > total:
                    start = max(0.0, total - self.CLIP_DURATION)
                self.log_line.emit(
                    f"▸ Clipping {self.CLIP_DURATION}s from {start:.1f}s...")
                self._trim_audio(self.audio_path, clip_path,
                                 start, self.CLIP_DURATION)
                self.progress.emit(15)
                sep = STEMSeparatorLogic(
                    self.config, progress_cb=lambda v: self.progress.emit(15 + int(v * 0.8)))
                stems = sep.separate(clip_path, out_dir)
                sys.stdout = old
                vocals_path = str(stems.get("vocals", ""))
                self.finished.emit(True, "Preview ready", vocals_path)
            except Exception as e:
                sys.stdout = old
                self.log_line.emit(f"ERROR: {e}")
                self.finished.emit(False, str(e), "")

        def cleanup(self):
            if self._tmp_dir and self._tmp_dir.exists():
                try:
                    shutil.rmtree(str(self._tmp_dir))
                except Exception:
                    pass

    class PreviewDialog(QDialog):
        def __init__(self, config: SeparationConfig, audio_path: Path, parent=None):
            super().__init__(parent)
            self.setWindowTitle("Preview — 10s Separation")
            self.setMinimumWidth(620)
            self.setStyleSheet(
                STYLE + f"QDialog{{background:{BG};border:1px solid {BORDER2};}}")
            self._worker: Optional[PreviewWorker] = None
            self._player: Optional[QMediaPlayer] = None
            self._audio_out: Optional[QAudioOutput] = None
            self._duration = 0
            self._seeking = False
            self._config = config
            self._audio_path = audio_path

            lay = QVBoxLayout(self)
            lay.setContentsMargins(28, 22, 28, 22)
            lay.setSpacing(14)

            title_row = QHBoxLayout()
            title_lbl = QLabel("PREVIEW")
            title_lbl.setStyleSheet(
                f"color:{CYAN};font-size:14px;font-weight:bold;letter-spacing:3px;")
            preset_lbl = QLabel(config.quality_preset.upper())
            preset_lbl.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
            self._close_btn = GlowButton("✕  Close")
            self._close_btn.clicked.connect(self._on_close)
            title_row.addWidget(title_lbl)
            title_row.addSpacing(12)
            title_row.addWidget(preset_lbl)
            title_row.addStretch()
            title_row.addWidget(self._close_btn)
            lay.addLayout(title_row)

            self._status = QLabel("Generating preview...")
            self._status.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
            lay.addWidget(self._status)

            self._prog = QProgressBar()
            self._prog.setFixedHeight(4)
            self._prog.setTextVisible(False)
            self._prog.setRange(0, 100)
            lay.addWidget(self._prog)

            self._wave = WaveformWidget(CYAN)
            self._wave.setVisible(False)
            self._wave.seek_requested.connect(self._on_wave_seek)
            lay.addWidget(self._wave)

            ctrl_row = QHBoxLayout()
            ctrl_row.setSpacing(10)
            self._play_btn = QPushButton("▶")
            self._play_btn.setFixedSize(40, 40)
            self._play_btn.setEnabled(False)
            self._play_btn.setCursor(
                QCursor(Qt.CursorShape.PointingHandCursor))
            self._play_btn.setStyleSheet(f"""
                QPushButton {{
                    background:{CYAN_DARK}; border:1px solid {CYAN};
                    border-radius:20px; color:{CYAN}; font-size:14px;
                }}
                QPushButton:hover   {{ background:{CYAN}; color:#000; }}
                QPushButton:pressed {{ background:{CYAN_DIM}; }}
                QPushButton:disabled {{ border-color:{BORDER2}; color:{TEXT_MUTED}; background:{BG2}; }}
            """)
            self._play_btn.clicked.connect(self._toggle_play)
            self._pos_lbl = QLabel("0:00")
            self._pos_lbl.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
            self._seek_sl = QSlider(Qt.Orientation.Horizontal)
            self._seek_sl.setRange(0, 1000)
            self._seek_sl.setEnabled(False)
            self._seek_sl.sliderPressed.connect(
                lambda: setattr(self, "_seeking", True))
            self._seek_sl.sliderReleased.connect(self._on_seek_release)
            self._seek_sl.setStyleSheet(f"""
                QSlider::groove:horizontal {{ background:{BORDER2}; height:3px; border-radius:1px; }}
                QSlider::handle:horizontal {{
                    background:{CYAN}; width:12px; height:12px; border-radius:6px; margin:-5px 0;
                }}
                QSlider::sub-page:horizontal {{ background:{CYAN}; border-radius:1px; }}
            """)
            self._dur_lbl = QLabel("0:00")
            self._dur_lbl.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
            ctrl_row.addWidget(self._play_btn)
            ctrl_row.addWidget(self._pos_lbl)
            ctrl_row.addWidget(self._seek_sl, stretch=1)
            ctrl_row.addWidget(self._dur_lbl)
            lay.addLayout(ctrl_row)

            self._log = QTextEdit()
            self._log.setReadOnly(True)
            self._log.setFixedHeight(90)
            self._log.setFont(QFont("Consolas", 10))
            lay.addWidget(self._log)

            self._start_worker()

        def _start_worker(self):
            self._worker = PreviewWorker(self._config, self._audio_path, self)
            self._worker.log_line.connect(self._on_log)
            self._worker.progress.connect(self._prog.setValue)
            self._worker.finished.connect(self._on_done)
            self._worker.start()

        def _on_log(self, text: str):
            self._log.append(f'<span style="color:{TEXT_DIM};">{text}</span>')
            self._status.setText(text[:80])

        def _on_done(self, ok: bool, msg: str, vocals_path: str):
            self._prog.setValue(100)
            if ok and vocals_path and Path(vocals_path).exists():
                self._status.setText("✓  Preview ready — vocals isolated")
                self._status.setStyleSheet(f"color:{SUCCESS};font-size:12px;")
                self._setup_player(vocals_path)
            else:
                self._status.setText(f"✗  {msg}")
                self._status.setStyleSheet(f"color:{ERROR_C};font-size:12px;")

        def _setup_player(self, path: str):
            self._player = QMediaPlayer()
            self._audio_out = QAudioOutput()
            self._player.setAudioOutput(self._audio_out)
            self._audio_out.setVolume(0.9)
            self._player.setSource(QUrl.fromLocalFile(path))
            self._player.positionChanged.connect(self._on_pos)
            self._player.durationChanged.connect(self._on_dur)
            self._player.playbackStateChanged.connect(self._on_state)
            bars = load_waveform_data(path, num_bars=250)
            self._wave.set_bars(bars)
            self._wave.setVisible(True)
            self._play_btn.setEnabled(True)
            self._seek_sl.setEnabled(True)

        @staticmethod
        def _fmt(ms: int) -> str:
            s = ms // 1000
            return f"{s // 60}:{s % 60:02d}"

        def _toggle_play(self):
            if not self._player:
                return
            if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self._player.pause()
            else:
                self._player.play()

        def _on_state(self, state):
            self._play_btn.setText(
                "⏸" if state == QMediaPlayer.PlaybackState.PlayingState else "▶")

        def _on_pos(self, pos: int):
            if not self._seeking:
                self._pos_lbl.setText(self._fmt(pos))
                if self._duration > 0:
                    frac = pos / self._duration
                    self._seek_sl.setValue(int(frac * 1000))
                    self._wave.set_progress(frac)

        def _on_dur(self, dur: int):
            self._duration = dur
            self._dur_lbl.setText(self._fmt(dur))

        def _on_seek_release(self):
            self._seeking = False
            if self._duration > 0:
                self._player.setPosition(
                    int(self._seek_sl.value() / 1000 * self._duration))

        def _on_wave_seek(self, frac: float):
            if self._player and self._duration > 0:
                self._player.setPosition(int(frac * self._duration))

        def _on_close(self):
            if self._player:
                self._player.stop()
                self._player.setSource(QUrl())
            if self._worker and self._worker.isRunning():
                self._worker.terminate()
                self._worker.wait()
            if self._worker:
                self._worker.cleanup()
            self.close()

        def closeEvent(self, e):
            self._on_close()
            super().closeEvent(e)

    class SeparationWorker(QThread):
        log_line = pyqtSignal(str)
        progress = pyqtSignal(int)
        finished = pyqtSignal(bool, str)
        stem_ready = pyqtSignal(str, str)

        def __init__(self, config, audio_path, output_dir):
            super().__init__()
            self.config = config
            self.audio_path = audio_path
            self.output_dir = output_dir

        def run(self):
            old = sys.stdout
            try:
                sig = self.log_line

                class Tee:
                    def __init__(self): self.buf = ""

                    def write(self, t):
                        self.buf += t
                        while "\n" in self.buf:
                            line, self.buf = self.buf.split("\n", 1)
                            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
                            if clean.strip():
                                sig.emit(clean)

                    def flush(self): pass
                sys.stdout = Tee()
                sep = STEMSeparatorLogic(
                    self.config, progress_cb=self.progress.emit)
                stems = sep.separate(self.audio_path, self.output_dir)
                sys.stdout = old
                for n, p in stems.items():
                    self.stem_ready.emit(n, str(p))
                self.finished.emit(True, "Separation complete")
            except Exception as e:
                sys.stdout = old
                self.log_line.emit(f"ERROR: {e}")
                self.log_line.emit(traceback.format_exc())
                self.finished.emit(False, str(e))

    class BatchWorker(QThread):
        log_line = pyqtSignal(str)
        progress = pyqtSignal(int)
        file_started = pyqtSignal(str, int, int)
        file_done = pyqtSignal(str, bool)
        finished = pyqtSignal(bool, str)

        def __init__(self, config: SeparationConfig, audio_paths: list, batch_dir: Path):
            super().__init__()
            self.config = config
            self.audio_paths = audio_paths
            self.batch_dir = batch_dir
            self._cancelled = False

        def cancel(self):
            self._cancelled = True

        def run(self):
            old = sys.stdout
            total = len(self.audio_paths)
            all_ok = True
            try:
                sig = self.log_line

                class Tee:
                    def __init__(self): self.buf = ""

                    def write(self, t):
                        self.buf += t
                        while "\n" in self.buf:
                            line, self.buf = self.buf.split("\n", 1)
                            clean = re.sub(r"\x1b\[[0-9;]*m", "", line)
                            if clean.strip():
                                sig.emit(clean)

                    def flush(self): pass
                sys.stdout = Tee()
                for idx, audio_path in enumerate(self.audio_paths):
                    if self._cancelled:
                        break
                    ap = Path(audio_path)
                    self.file_started.emit(ap.name, idx + 1, total)
                    self.log_line.emit(f"\n{'─'*50}")
                    self.log_line.emit(f"  [{idx+1}/{total}]  {ap.name}")
                    self.log_line.emit(f"{'─'*50}")
                    out_dir = self.batch_dir / ap.stem

                    def _prog(v, _idx=idx, _total=total):
                        overall = int((_idx / _total) * 100 + v / _total)
                        self.progress.emit(overall)
                    try:
                        sep = STEMSeparatorLogic(
                            self.config, progress_cb=_prog)
                        stems = sep.separate(ap, out_dir)
                        _songs_json_append(
                            ap.stem, out_dir, stems,
                            preset=self.config.quality_preset,
                            model=self.config.model,
                            backend_desc="batch",
                        )
                        self.file_done.emit(ap.name, True)
                    except Exception as e:
                        self.log_line.emit(f"ERROR on {ap.name}: {e}")
                        self.file_done.emit(ap.name, False)
                        all_ok = False
                sys.stdout = old
                self.progress.emit(100)
                msg = "Batch complete" if not self._cancelled else "Batch cancelled"
                self.finished.emit(all_ok and not self._cancelled, msg)
            except Exception as e:
                sys.stdout = old
                self.log_line.emit(f"BATCH ERROR: {e}")
                self.finished.emit(False, str(e))

    def _sec_lbl(text: str) -> QLabel:
        l = QLabel(text)
        l.setStyleSheet(
            f"color:{TEXT_MUTED};font-size:10px;letter-spacing:3px;background:transparent;")
        return l

    class GlowButton(QPushButton):
        def __init__(self, text, parent=None, primary=False, danger=False, accent=None):
            super().__init__(text, parent)
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            c = accent if accent else (ERROR_C if danger else CYAN)
            dim = "#cc3333" if danger else (CYAN_DIM if not accent else accent)
            glow = "#ff444433" if danger else (
                CYAN_GLOW if not accent else f"{accent}33")
            dark = "#3d0000" if danger else (
                CYAN_DARK if not accent else f"{accent}22")
            if primary:
                self.setStyleSheet(f"""
                    QPushButton {{
                        background:transparent; border:1px solid {c}; border-radius:6px;
                        color:{c}; font-size:13px; font-weight:bold;
                        padding:9px 28px; letter-spacing:2px;
                    }}
                    QPushButton:hover   {{ background:{glow}; color:#fff; }}
                    QPushButton:pressed {{ background:{dark}; }}
                    QPushButton:disabled {{ border-color:{BORDER2}; color:{TEXT_MUTED}; }}
                """)
            else:
                self.setStyleSheet(f"""
                    QPushButton {{
                        background:transparent; border:1px solid {BORDER2}; border-radius:6px;
                        color:{TEXT_DIM}; font-size:12px; padding:6px 14px;
                    }}
                    QPushButton:hover   {{ border-color:{dim}; color:{TEXT}; }}
                    QPushButton:pressed {{ background:{BG2}; }}
                """)

    class PresetCard(QFrame):
        clicked = pyqtSignal(str)
        INFO = {
            "fast":         ("FAST",         "htdemucs · 1 shift",      "Quick, good quality"),
            "fine-tuned":   ("FINE-TUNED",   "htdemucs_ft · 2 shifts",  "Better accuracy, moderate speed"),
            "professional": ("PROFESSIONAL", "MDX + Demucs · 4 shifts", "Ensemble hybrid pipeline"),
            "max":          ("MAX",          "Full ensemble + cleanup",  "Best quality · configurable"),
        }

        def __init__(self, key: str, parent=None):
            super().__init__(parent)
            self.key = key
            self.active = False
            title, sub, desc = self.INFO[key]
            self.setFixedHeight(84)
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            lay = QVBoxLayout(self)
            lay.setContentsMargins(15, 11, 15, 11)
            lay.setSpacing(5)
            row = QHBoxLayout()
            row.setSpacing(0)
            self.t = QLabel(title)
            self.t.setStyleSheet(
                f"color:{CYAN};font-size:12px;font-weight:bold;letter-spacing:2px;")
            self.s = QLabel(sub)
            self.s.setStyleSheet(f"color:{TEXT_MUTED};font-size:10px;")
            row.addWidget(self.t)
            row.addStretch()
            row.addWidget(self.s)
            self.d = QLabel(desc)
            self.d.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
            lay.addLayout(row)
            lay.addWidget(self.d)
            self._refresh()

        def _refresh(self):
            c = CYAN if self.active else BORDER
            bg = "#0d1a1e" if self.active else BG2
            self.setStyleSheet(
                f"PresetCard{{background:{bg};border:1px solid {c};border-radius:8px;}}")

        def set_active(self, v: bool):
            self.active = v
            self._refresh()
            self.t.setStyleSheet(
                f"color:{'#ffffff' if v else CYAN};font-size:12px;font-weight:bold;letter-spacing:2px;")

        def mousePressEvent(self, e):
            self.clicked.emit(self.key)
            super().mousePressEvent(e)

    class PulsingDot(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.setFixedSize(10, 10)
            self._r = 3.0
            self._anim = QPropertyAnimation(self, b"dot_r")
            self._anim.setStartValue(2.0)
            self._anim.setEndValue(4.5)
            self._anim.setDuration(800)
            self._anim.setLoopCount(-1)
            self._anim.setEasingCurve(QEasingCurve.Type.SineCurve)

        def start(self): self._anim.start()
        def stop(self):  self._anim.stop(); self._r = 3.0; self.update()
        def get_r(self): return self._r
        def set_r(self, v): self._r = v; self.update()
        dot_r = pyqtProperty(float, get_r, set_r)

        def paintEvent(self, _):
            p = QPainter(self)
            p.setRenderHint(QPainter.RenderHint.Antialiasing)
            p.setPen(Qt.PenStyle.NoPen)
            c = QColor(CYAN)
            c.setAlpha(180)
            p.setBrush(QBrush(c))
            cx, cy, r = self.width()//2, self.height()//2, self._r
            p.drawEllipse(int(cx-r), int(cy-r), int(r*2), int(r*2))

    STEM_COLOURS = {
        "vocals":       "#00e5ff",
        "bass":         "#00c8e0",
        "drums":        "#00aabf",
        "other":        "#008fa0",
        "guitar":       "#00d4a8",
        "piano":        "#00b890",
        "instrumental": "#007280",
    }
    STEM_ICONS = {
        "vocals": "◈", "bass": "◉", "drums": "◎", "other": "◐",
        "guitar": "◍", "piano": "◌", "instrumental": "◑",
    }

    class WaveformLoader(QThread):
        done = pyqtSignal(list)

        def __init__(self, path: str, num_bars: int = 350):
            super().__init__()
            self._path = path
            self._num_bars = num_bars

        def run(self):
            bars = load_waveform_data(self._path, self._num_bars)
            self.done.emit(bars)

    class StemPlayer(QFrame):
        def __init__(self, stem_name: str, file_path: str, parent=None):
            super().__init__(parent)
            self.stem_name = stem_name
            self.file_path = file_path
            self._seeking = False
            self._duration = 0
            colour = STEM_COLOURS.get(stem_name, CYAN)
            self.setStyleSheet(
                f"StemPlayer{{background:{BG2};border:1px solid {BORDER};border-radius:12px;}}")
            self._player = QMediaPlayer()
            self._audio = QAudioOutput()
            self._player.setAudioOutput(self._audio)
            self._audio.setVolume(0.9)
            self._player.setSource(QUrl.fromLocalFile(str(file_path)))
            self._player.positionChanged.connect(self._on_pos)
            self._player.durationChanged.connect(self._on_dur)
            self._player.playbackStateChanged.connect(self._on_state)

            root = QVBoxLayout(self)
            root.setContentsMargins(22, 18, 22, 18)
            root.setSpacing(14)

            top = QHBoxLayout()
            top.setSpacing(12)
            icon_l = QLabel(STEM_ICONS.get(stem_name, "▸"))
            icon_l.setStyleSheet(f"color:{colour};font-size:22px;")
            icon_l.setFixedWidth(26)
            name_l = QLabel(stem_name.upper())
            name_l.setStyleSheet(
                f"color:{colour};font-size:13px;font-weight:bold;letter-spacing:3px;")
            p = Path(file_path)
            file_l = QLabel(p.name)
            file_l.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
            mb = p.stat().st_size / 1_048_576 if p.exists() else 0
            sz_l = QLabel(f"{mb:.1f} MB")
            sz_l.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
            sz_l.setAlignment(Qt.AlignmentFlag.AlignRight |
                              Qt.AlignmentFlag.AlignVCenter)
            top.addWidget(icon_l)
            top.addWidget(name_l)
            top.addWidget(file_l, stretch=1)
            top.addWidget(sz_l)
            root.addLayout(top)

            self._wave = WaveformWidget(colour=colour)
            self._wave.seek_requested.connect(self._on_wave_seek)
            root.addWidget(self._wave)

            ctrl = QHBoxLayout()
            ctrl.setSpacing(10)
            self._play_btn = QPushButton("▶")
            self._play_btn.setFixedSize(38, 38)
            self._play_btn.setCursor(
                QCursor(Qt.CursorShape.PointingHandCursor))
            self._play_btn.setStyleSheet(f"""
                QPushButton {{
                    background:{CYAN_DARK}; border:1px solid {colour};
                    border-radius:19px; color:{colour}; font-size:14px;
                }}
                QPushButton:hover   {{ background:{colour}; color:#000; }}
                QPushButton:pressed {{ background:{CYAN_DIM}; }}
            """)
            self._play_btn.clicked.connect(self._toggle)
            self._pos_l = QLabel("0:00")
            self._pos_l.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
            self._pos_l.setFixedWidth(38)
            self._seek = QSlider(Qt.Orientation.Horizontal)
            self._seek.setRange(0, 1000)
            self._seek.sliderPressed.connect(
                lambda: setattr(self, "_seeking", True))
            self._seek.sliderReleased.connect(self._on_seek_release)
            self._seek.setStyleSheet(f"""
                QSlider::groove:horizontal {{ background:{BORDER2}; height:3px; border-radius:1px; }}
                QSlider::handle:horizontal {{
                    background:{colour}; width:13px; height:13px; border-radius:6px; margin:-5px 0;
                }}
                QSlider::sub-page:horizontal {{ background:{colour}; border-radius:1px; }}
            """)
            self._dur_l = QLabel("0:00")
            self._dur_l.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
            self._dur_l.setFixedWidth(38)
            self._dur_l.setAlignment(
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            ctrl.addWidget(self._play_btn)
            ctrl.addWidget(self._pos_l)
            ctrl.addWidget(self._seek, stretch=1)
            ctrl.addWidget(self._dur_l)
            root.addLayout(ctrl)

            self._wave_loader = WaveformLoader(file_path, 350)
            self._wave_loader.done.connect(self._wave.set_bars)
            self._wave_loader.start()

        @staticmethod
        def _fmt(ms: int) -> str:
            s = ms // 1000
            return f"{s//60}:{s % 60:02d}"

        def _toggle(self):
            if self._player.playbackState() == QMediaPlayer.PlaybackState.PlayingState:
                self._player.pause()
            else:
                self._player.play()

        def _on_state(self, state):
            self._play_btn.setText(
                "⏸" if state == QMediaPlayer.PlaybackState.PlayingState else "▶")

        def _on_pos(self, pos: int):
            if not self._seeking:
                self._pos_l.setText(self._fmt(pos))
                if self._duration > 0:
                    frac = pos / self._duration
                    self._seek.setValue(int(frac * 1000))
                    self._wave.set_progress(frac)

        def _on_dur(self, dur: int):
            self._duration = dur
            self._dur_l.setText(self._fmt(dur))

        def _on_seek_release(self):
            self._seeking = False
            if self._duration > 0:
                self._player.setPosition(
                    int(self._seek.value() / 1000 * self._duration))

        def _on_wave_seek(self, frac: float):
            if self._duration > 0:
                self._player.setPosition(int(frac * self._duration))

        def stop_playback(self): self._player.stop()

        def cleanup(self):
            self._player.stop()
            self._player.setSource(QUrl())

    class SongCard(QFrame):
        clicked = pyqtSignal(dict)

        def __init__(self, song_name: str, out_dir: str, n_stems: int,
                     preset: str = "", date: str = "", record: dict = None, parent=None):
            super().__init__(parent)
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self.setFixedHeight(120)
            self._hover = False
            self._record = record if record is not None else {
                "song_name": song_name, "out_dir": out_dir,
                "stems": {}, "preset": preset, "date": date,
            }
            self._refresh()
            outer = QHBoxLayout(self)
            outer.setContentsMargins(0, 0, 0, 0)
            outer.setSpacing(0)
            bar = QFrame()
            bar.setFixedWidth(4)
            bar.setStyleSheet(f"background:{CYAN};border-radius:2px;")
            outer.addWidget(bar)
            inner = QHBoxLayout()
            inner.setContentsMargins(20, 0, 20, 0)
            inner.setSpacing(14)
            deco = QLabel("▐▌▐")
            deco.setStyleSheet(
                f"color:{CYAN_DIM};font-size:22px;letter-spacing:2px;")
            text_col = QVBoxLayout()
            text_col.setSpacing(4)
            title = QLabel(song_name.upper())
            title.setStyleSheet(
                f"color:{TEXT};font-size:17px;font-weight:bold;letter-spacing:3px;")
            meta_parts = []
            if preset:
                meta_parts.append(preset.upper())
            if date:
                try:
                    dt = datetime.datetime.fromisoformat(date)
                    meta_parts.append(dt.strftime("%d %b %Y  %H:%M"))
                except Exception:
                    meta_parts.append(date[:16])
            meta_lbl = QLabel("  ·  ".join(meta_parts)
                              if meta_parts else out_dir)
            meta_lbl.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
            path_lbl = QLabel(out_dir)
            path_lbl.setStyleSheet(f"color:{TEXT_MUTED};font-size:10px;")
            text_col.addStretch()
            text_col.addWidget(title)
            text_col.addWidget(meta_lbl)
            text_col.addWidget(path_lbl)
            text_col.addStretch()
            right = QVBoxLayout()
            right.setSpacing(6)
            badge = QLabel(f"{n_stems}  stems")
            badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
            badge.setStyleSheet(f"""
                background:{CYAN_DARK}; color:{CYAN}; font-size:11px;
                font-weight:bold; border:1px solid {CYAN_DIM}; border-radius:10px;
                padding:3px 10px;
            """)
            arrow = QLabel("›")
            arrow.setStyleSheet(f"color:{TEXT_MUTED};font-size:24px;")
            arrow.setAlignment(Qt.AlignmentFlag.AlignRight |
                               Qt.AlignmentFlag.AlignVCenter)
            right.addStretch()
            right.addWidget(badge, alignment=Qt.AlignmentFlag.AlignRight)
            right.addWidget(arrow, alignment=Qt.AlignmentFlag.AlignRight)
            right.addStretch()
            inner.addWidget(deco)
            inner.addLayout(text_col, stretch=1)
            inner.addLayout(right)
            outer.addLayout(inner, stretch=1)

        def _refresh(self):
            if self._hover:
                self.setStyleSheet(
                    f"SongCard{{background:#0d1a1e;border:1px solid {CYAN};border-radius:12px;}}")
            else:
                self.setStyleSheet(
                    f"SongCard{{background:{BG2};border:1px solid {BORDER2};border-radius:12px;}}")

        def enterEvent(self, e):  self._hover = True;  self._refresh(
        ); super().enterEvent(e)

        def leaveEvent(self, e):  self._hover = False; self._refresh(
        ); super().leaveEvent(e)
        def mousePressEvent(self, e): self.clicked.emit(
            self._record); super().mousePressEvent(e)

    class StemRowCard(QFrame):
        clicked = pyqtSignal(str, str)

        def __init__(self, stem_name: str, file_path: str, parent=None):
            super().__init__(parent)
            self.stem_name = stem_name
            self.file_path = file_path
            self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
            self._colour = STEM_COLOURS.get(stem_name, CYAN)
            self.setFixedHeight(72)
            self._hover = False
            self._refresh()
            lay = QHBoxLayout(self)
            lay.setContentsMargins(0, 0, 16, 0)
            lay.setSpacing(0)
            stripe = QFrame()
            stripe.setFixedWidth(3)
            stripe.setStyleSheet(
                f"background:{self._colour};border-radius:1px;")
            lay.addWidget(stripe)
            lay.addSpacing(16)
            icon = QLabel(STEM_ICONS.get(stem_name, "▸"))
            icon.setStyleSheet(f"color:{self._colour};font-size:20px;")
            icon.setFixedWidth(26)
            name_lbl = QLabel(stem_name.upper())
            name_lbl.setStyleSheet(
                f"color:{self._colour};font-size:13px;font-weight:bold;letter-spacing:2px;")
            name_lbl.setFixedWidth(120)
            p = Path(file_path)
            file_lbl = QLabel(p.name)
            file_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
            mb = p.stat().st_size / 1_048_576 if p.exists() else 0
            sz_lbl = QLabel(f"{mb:.1f} MB")
            sz_lbl.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
            sz_lbl.setFixedWidth(54)
            sz_lbl.setAlignment(Qt.AlignmentFlag.AlignRight |
                                Qt.AlignmentFlag.AlignVCenter)
            play_hint = QLabel("▶  play")
            play_hint.setStyleSheet(
                f"color:{self._colour};font-size:11px;border:1px solid {self._colour}33;"
                f"border-radius:9px;padding:2px 10px;")
            arrow = QLabel("›")
            arrow.setStyleSheet(f"color:{TEXT_MUTED};font-size:20px;")
            arrow.setFixedWidth(14)
            lay.addWidget(icon)
            lay.addSpacing(4)
            lay.addWidget(name_lbl)
            lay.addWidget(file_lbl, stretch=1)
            lay.addWidget(sz_lbl)
            lay.addSpacing(12)
            lay.addWidget(play_hint)
            lay.addSpacing(8)
            lay.addWidget(arrow)

        def _refresh(self):
            c = self._colour if self._hover else BORDER
            bg = "#0d1a1e" if self._hover else BG2
            self.setStyleSheet(
                f"StemRowCard{{background:{bg};border:1px solid {c};border-radius:10px;}}")

        def enterEvent(self, e): self._hover = True;  self._refresh(
        ); super().enterEvent(e)

        def leaveEvent(self, e): self._hover = False; self._refresh(
        ); super().leaveEvent(e)

        def mousePressEvent(self, e):
            self.clicked.emit(self.stem_name, self.file_path)
            super().mousePressEvent(e)

    class ModelFuseDialog(QDialog):
        def __init__(self, models_dict: Dict[str, Any], parent=None):
            super().__init__(parent)
            self.setWindowTitle("Fuse Models")
            self.setMinimumWidth(500)
            self.setStyleSheet(f"background:{BG}; border:1px solid {BORDER2};")
            self.models_dict = models_dict
            self.base_models = {k: v for k, v in models_dict.items()
                                if v.get("type") == "base" and not v.get("builtin", False)}

            lay = QVBoxLayout(self)
            lay.setContentsMargins(24, 24, 24, 24)
            lay.setSpacing(16)

            title = QLabel("FUSE TWO MODELS")
            title.setStyleSheet(
                f"color:{CYAN}; font-size:16px; font-weight:bold;")
            lay.addWidget(title)

            info = QLabel(
                "Select two base models to blend their vocal outputs into a new fused model.")
            info.setStyleSheet(f"color:{TEXT_DIM}; font-size:12px;")
            info.setWordWrap(True)
            lay.addWidget(info)

            a_row = QHBoxLayout()
            a_label = QLabel("Model A")
            a_label.setStyleSheet(f"color:{TEXT};")
            a_label.setFixedWidth(80)
            self.model_a_combo = QComboBox()
            for key, m in self.base_models.items():
                self.model_a_combo.addItem(m["name"], key)
            a_row.addWidget(a_label)
            a_row.addWidget(self.model_a_combo, stretch=1)
            lay.addLayout(a_row)

            b_row = QHBoxLayout()
            b_label = QLabel("Model B")
            b_label.setStyleSheet(f"color:{TEXT};")
            b_label.setFixedWidth(80)
            self.model_b_combo = QComboBox()
            for key, m in self.base_models.items():
                self.model_b_combo.addItem(m["name"], key)
            b_row.addWidget(b_label)
            b_row.addWidget(self.model_b_combo, stretch=1)
            lay.addLayout(b_row)

            name_row = QHBoxLayout()
            name_label = QLabel("Fused name")
            name_label.setStyleSheet(f"color:{TEXT};")
            name_label.setFixedWidth(80)
            self.name_edit = QLineEdit()
            self.name_edit.setPlaceholderText("e.g. MyFusion")
            name_row.addWidget(name_label)
            name_row.addWidget(self.name_edit, stretch=1)
            lay.addLayout(name_row)

            btn_box = QDialogButtonBox(
                QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
            btn_box.accepted.connect(self.accept)
            btn_box.rejected.connect(self.reject)
            lay.addWidget(btn_box)

        def get_result(self):
            return {
                "key": self.name_edit.text().strip().replace(" ", "_"),
                "name": self.name_edit.text().strip(),
                "model_a": self.model_a_combo.currentData(),
                "model_b": self.model_b_combo.currentData(),
            }

    AVAILABLE_DOWNLOAD_MODELS = [
        ("MDX23C-8KFFT-InstVoc_HQ",    "MDX23C-8KFFT-InstVoc_HQ.ckpt",
         "MDX-C",  "Highest accuracy vocal isolation"),
        ("Kim_Vocal_2",                 "Kim_Vocal_2.onnx",
         "MDX",    "Clean, natural vocal tone"),
        ("UVR-MDX-NET-Voc_FT",         "UVR-MDX-NET-Voc_FT.onnx",
         "MDX",    "Balanced vocals, fast"),
        ("UVR-MDX-NET-Inst_HQ_4",      "UVR-MDX-NET-Inst_HQ_4.onnx",
         "MDX",    "High-quality instrumental"),
        ("UVR-MDX-NET-Inst_1",         "UVR-MDX-NET-Inst_1.onnx",
         "MDX",    "Clean instrumental separation"),
        ("UVR_MDXNET_KARA_2",          "UVR_MDXNET_KARA_2.onnx",
         "MDX",    "Karaoke / backing track"),
        ("Reverb_HQ_By_FoxJoy",        "Reverb_HQ_By_FoxJoy.onnx",
         "MDX",    "Reverb removal (high quality)"),
        ("UVR-DeEcho-DeReverb",        "UVR-DeEcho-DeReverb.pth",
         "VR",     "Echo + reverb removal"),
        ("UVR-DeNoise",                "UVR-DeNoise.pth",
         "VR",     "Background noise removal"),
        ("UVR_MDXNET_9482",            "UVR_MDXNET_9482.onnx",
         "MDX",    "General purpose separation"),
        ("UVR-MDX-NET-Inst_Main",      "UVR-MDX-NET-Inst_Main.onnx",
         "MDX",    "Main instrumental model"),
        ("5_HP-Karaoke-UVR",           "5_HP-Karaoke-UVR.pth",
         "VR",     "Karaoke vocal removal"),
        ("UVR-BVE-4B_SN-44100-1",     "UVR-BVE-4B_SN-44100-1.pth",
         "VR",     "Backing vocal extraction"),
        ("Kim_Inst",                   "Kim_Inst.onnx",
         "MDX",    "Kim instrumental variant"),
        ("UVR-MDX-NET_Inst_187_beta",  "UVR-MDX-NET_Inst_187_beta.onnx",
         "MDX",    "Instrumental beta model"),
        ("mel_band_roformer_karaoke",  "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt",
         "MDX-C", "Roformer karaoke model"),
    ]

    class HFDownloadWorker(QThread):
        log_line = pyqtSignal(str)
        progress = pyqtSignal(int)
        finished = pyqtSignal(bool, str)

        def __init__(self, repo_id: str, filename: str, dest_dir: Path):
            super().__init__()
            self.repo_id = repo_id.strip("/")
            self.filename = filename
            self.dest_dir = dest_dir
            self._cancelled = False

        def cancel(self):
            self._cancelled = True

        def run(self):
            import urllib.request
            import urllib.error
            dest_path = self.dest_dir / self.filename
            if dest_path.exists():
                self.log_line.emit(f"Already exists: {self.filename}")
                self.finished.emit(True, str(dest_path))
                return
            url = f"https://huggingface.co/{self.repo_id}/resolve/main/{self.filename}"
            self.log_line.emit(
                f"Downloading {self.filename} from {self.repo_id}...")
            try:
                req = urllib.request.Request(
                    url, headers={"User-Agent": "StemLab/1.0"})
                with urllib.request.urlopen(req) as response:
                    total_size = int(response.headers.get('Content-Length', 0))
                    downloaded = 0
                    chunk_size = 8192
                    with open(dest_path, 'wb') as f:
                        while True:
                            if self._cancelled:
                                dest_path.unlink(missing_ok=True)
                                self.log_line.emit("Download cancelled.")
                                self.finished.emit(False, "Cancelled")
                                return
                            chunk = response.read(chunk_size)
                            if not chunk:
                                break
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size:
                                pct = int(downloaded / total_size * 100)
                                self.progress.emit(pct)
                self.log_line.emit(f"✓ Saved to {dest_path}")
                self.finished.emit(True, str(dest_path))
            except urllib.error.HTTPError as e:
                self.log_line.emit(f"HTTP error: {e.code} {e.reason}")
                dest_path.unlink(missing_ok=True)
                self.finished.emit(False, f"HTTP {e.code}")
            except Exception as e:
                self.log_line.emit(f"Error: {e}")
                dest_path.unlink(missing_ok=True)
                self.finished.emit(False, str(e))

    class HuggingFaceImportDialog(QDialog):
        def __init__(self, models_page, parent=None):
            super().__init__(parent)
            self.models_page = models_page
            self.setWindowTitle("Import from Hugging Face")
            self.setMinimumWidth(520)
            self.setStyleSheet(
                STYLE + f"QDialog{{background:{BG};border:1px solid {BORDER2};}}")
            self._worker = None
            self._dest_dir = _models_dir()
            self._build_ui()

        def _build_ui(self):
            lay = QVBoxLayout(self)
            lay.setContentsMargins(28, 24, 28, 24)
            lay.setSpacing(16)

            title = QLabel("IMPORT FROM HUGGING FACE")
            title.setStyleSheet(
                f"color:{CYAN};font-size:14px;font-weight:bold;letter-spacing:3px;")
            lay.addWidget(title)

            info = QLabel(
                "Enter the Hugging Face repository ID and filename of the model you want to download.")
            info.setStyleSheet(f"color:{TEXT_DIM};font-size:11px;")
            info.setWordWrap(True)
            lay.addWidget(info)

            repo_row = QHBoxLayout()
            repo_lbl = QLabel("Repository:")
            repo_lbl.setStyleSheet(f"color:{TEXT};font-size:12px;")
            repo_lbl.setFixedWidth(80)
            self.repo_edit = QLineEdit()
            self.repo_edit.setPlaceholderText(
                "e.g. user/repository")
            repo_row.addWidget(repo_lbl)
            repo_row.addWidget(self.repo_edit, stretch=1)
            lay.addLayout(repo_row)

            file_row = QHBoxLayout()
            file_lbl = QLabel("Filename:")
            file_lbl.setStyleSheet(f"color:{TEXT};font-size:12px;")
            file_lbl.setFixedWidth(80)
            self.file_edit = QLineEdit()
            self.file_edit.setPlaceholderText("e.g. Kim_Vocal_2.onnx")
            file_row.addWidget(file_lbl)
            file_row.addWidget(self.file_edit, stretch=1)
            lay.addLayout(file_row)

            self._status_lbl = QLabel("")
            self._status_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
            lay.addWidget(self._status_lbl)

            self._prog = QProgressBar()
            self._prog.setFixedHeight(4)
            self._prog.setTextVisible(False)
            self._prog.setRange(0, 100)
            self._prog.setVisible(False)
            lay.addWidget(self._prog)

            self._log = QTextEdit()
            self._log.setReadOnly(True)
            self._log.setFixedHeight(90)
            self._log.setFont(QFont("Consolas", 10))
            self._log.setVisible(False)
            lay.addWidget(self._log)

            btn_row = QHBoxLayout()
            btn_row.addStretch()
            self._cancel_btn = GlowButton("✕  Cancel", danger=True)
            self._cancel_btn.setVisible(False)
            self._cancel_btn.clicked.connect(self._on_cancel)
            self._close_btn = GlowButton("Close")
            self._close_btn.clicked.connect(self.close)
            self._dl_btn = GlowButton("⬇  DOWNLOAD", primary=True)
            self._dl_btn.clicked.connect(self._start_download)
            btn_row.addWidget(self._cancel_btn)
            btn_row.addWidget(self._close_btn)
            btn_row.addWidget(self._dl_btn)
            lay.addLayout(btn_row)

        def _start_download(self):
            repo = self.repo_edit.text().strip()
            filename = self.file_edit.text().strip()
            if not repo or not filename:
                self._status_lbl.setText(
                    "Please enter both repository and filename.")
                return
            self._dl_btn.setEnabled(False)
            self._close_btn.setEnabled(False)
            self._cancel_btn.setVisible(True)
            self._prog.setVisible(True)
            self._log.setVisible(True)
            self._log.clear()
            self._status_lbl.setText("Downloading...")
            self._worker = HFDownloadWorker(repo, filename, self._dest_dir)
            self._worker.log_line.connect(self._on_log)
            self._worker.progress.connect(self._prog.setValue)
            self._worker.finished.connect(self._on_done)
            self._worker.start()

        def _on_log(self, text: str):
            self._log.append(f'<span style="color:{TEXT_DIM};">{text}</span>')

        def _on_done(self, ok: bool, msg: str):
            self._prog.setValue(100 if ok else 0)
            self._cancel_btn.setVisible(False)
            self._close_btn.setEnabled(True)
            self._dl_btn.setEnabled(True)
            if ok:
                self._status_lbl.setText("✓ Download complete")
                self._status_lbl.setStyleSheet(
                    f"color:{SUCCESS};font-size:12px;")
                if hasattr(self.models_page, "refresh_list"):
                    self.models_page.refresh_list()
                if hasattr(self.models_page, "main_window") and hasattr(self.models_page.main_window, "_populate_mdx_combo"):
                    self.models_page.main_window._populate_mdx_combo()
            else:
                self._status_lbl.setText(f"✗ Download failed: {msg}")
                self._status_lbl.setStyleSheet(
                    f"color:{ERROR_C};font-size:12px;")

        def _on_cancel(self):
            if self._worker and self._worker.isRunning():
                self._worker.cancel()
                self._worker.terminate()
                self._worker.wait()
            self._cancel_btn.setVisible(False)
            self._close_btn.setEnabled(True)
            self._dl_btn.setEnabled(True)
            self._status_lbl.setText("Download cancelled.")

        def closeEvent(self, e):
            self._on_cancel()
            super().closeEvent(e)

    class ModelDownloadWorker(QThread):
        log_line = pyqtSignal(str)
        progress = pyqtSignal(int)
        finished = pyqtSignal(bool, str)

        def __init__(self, model_filename: str, dest_dir: Path):
            super().__init__()
            self.model_filename = model_filename
            self.dest_dir = dest_dir
            self._cancelled = False

        def cancel(self):
            self._cancelled = True

        def run(self):
            dest_path = self.dest_dir / self.model_filename
            if dest_path.exists():
                self.log_line.emit(f"Already exists: {self.model_filename}")
                self.finished.emit(True, str(dest_path))
                return
            try:
                from audio_separator.separator import Separator
                self.log_line.emit(f"Downloading {self.model_filename}...")
                sep = Separator(model_file_dir=str(
                    self.dest_dir), output_format="wav")
                sep.load_model(model_filename=self.model_filename)
                downloaded = self.dest_dir / self.model_filename
                if downloaded.exists():
                    self.log_line.emit(f"✓ Saved to {dest_path}")
                    self.finished.emit(True, str(dest_path))
                else:
                    import glob
                    import shutil as _sh
                    tmp_dir = Path("/tmp/audio_separator_models")
                    matches = list(tmp_dir.glob(self.model_filename)
                                   ) if tmp_dir.exists() else []
                    if matches:
                        _sh.copy2(str(matches[0]), str(dest_path))
                        self.log_line.emit(f"✓ Saved to {dest_path}")
                        self.finished.emit(True, str(dest_path))
                    else:
                        self.finished.emit(
                            False, "File not found after download")
            except Exception as e:
                self.log_line.emit(f"Error: {e}")
                self.finished.emit(False, str(e))

    class ModelDownloadDialog(QDialog):
        def __init__(self, models_page, parent=None):
            super().__init__(parent)
            self.models_page = models_page
            self.setWindowTitle("Download Models")
            self.setMinimumSize(720, 560)
            self.setStyleSheet(
                STYLE + f"QDialog{{background:{BG};border:1px solid {BORDER2};}}")
            self._worker = None
            self._downloading = False
            self._queue = []
            self._current_idx = 0
            self._total = 0
            self._dest_dir = _models_dir()
            self._existing = {f.name for f in self._dest_dir.iterdir(
            )} if self._dest_dir.exists() else set()
            self._checkboxes = {}
            self._build_ui()

        def _build_ui(self):
            lay = QVBoxLayout(self)
            lay.setContentsMargins(28, 24, 28, 24)
            lay.setSpacing(14)

            hdr = QHBoxLayout()
            title = QLabel("DOWNLOAD MODELS")
            title.setStyleSheet(
                f"color:{CYAN};font-size:14px;font-weight:bold;letter-spacing:3px;")
            dest_lbl = QLabel(str(self._dest_dir))
            dest_lbl.setStyleSheet(f"color:{TEXT_MUTED};font-size:10px;")
            hdr.addWidget(title)
            hdr.addStretch()
            hdr.addWidget(dest_lbl)
            lay.addLayout(hdr)

            sel_row = QHBoxLayout()
            sel_all = QPushButton("Select All")
            sel_none = QPushButton("Select None")
            for b in (sel_all, sel_none):
                b.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                b.setStyleSheet(f"""
                    QPushButton {{
                        background:transparent; border:1px solid {BORDER2}; border-radius:5px;
                        color:{TEXT_DIM}; font-size:11px; padding:4px 12px;
                    }}
                    QPushButton:hover {{ border-color:{CYAN_DIM}; color:{TEXT}; }}
                """)
            sel_all.clicked.connect(lambda: [cb.setChecked(
                True) for cb in self._checkboxes.values()])
            sel_none.clicked.connect(lambda: [cb.setChecked(
                False) for cb in self._checkboxes.values()])
            sel_row.addWidget(sel_all)
            sel_row.addWidget(sel_none)
            sel_row.addStretch()
            lay.addLayout(sel_row)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet(
                "background:transparent;border:1px solid " + BORDER + ";border-radius:6px;")
            content = QWidget()
            content.setStyleSheet(f"background:{BG2};")
            grid = QVBoxLayout(content)
            grid.setContentsMargins(12, 10, 12, 10)
            grid.setSpacing(4)

            type_order = ["MDX-C", "MDX", "VR"]
            grouped = {}
            for entry in AVAILABLE_DOWNLOAD_MODELS:
                t = entry[2]
                grouped.setdefault(t, []).append(entry)

            for t in type_order:
                if t not in grouped:
                    continue
                type_hdr = QLabel(f"── {t} ──")
                type_hdr.setStyleSheet(
                    f"color:{TEXT_MUTED};font-size:10px;letter-spacing:2px;padding:6px 0 2px 0;")
                grid.addWidget(type_hdr)
                for key, filename, mtype, desc in grouped[t]:
                    row = QFrame()
                    already = filename in self._existing
                    row.setStyleSheet(
                        f"background:{'#0a1a0a' if already else BG3};border:1px solid "
                        f"{'#1a3a1a' if already else BORDER};border-radius:6px;"
                    )
                    row.setFixedHeight(52)
                    rl = QHBoxLayout(row)
                    rl.setContentsMargins(12, 0, 12, 0)
                    rl.setSpacing(10)

                    cb = QCheckBox()
                    cb.setChecked(not already)
                    cb.setEnabled(not already)
                    rl.addWidget(cb)
                    self._checkboxes[filename] = cb

                    name_col = QVBoxLayout()
                    name_col.setSpacing(1)
                    nl = QLabel(key)
                    nl.setStyleSheet(
                        f"color:{'#4a8a4a' if already else TEXT};font-size:12px;font-weight:bold;")
                    dl = QLabel(desc)
                    dl.setStyleSheet(f"color:{TEXT_MUTED};font-size:10px;")
                    name_col.addWidget(nl)
                    name_col.addWidget(dl)
                    rl.addLayout(name_col, stretch=1)

                    fn_lbl = QLabel("✓ installed" if already else filename)
                    fn_lbl.setStyleSheet(
                        f"color:{'#4a8a4a' if already else TEXT_MUTED};font-size:10px;"
                        f"{'font-weight:bold;' if already else ''}"
                    )
                    fn_lbl.setAlignment(
                        Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
                    rl.addWidget(fn_lbl)
                    grid.addWidget(row)

            grid.addStretch()
            scroll.setWidget(content)
            lay.addWidget(scroll, stretch=1)

            self._status_lbl = QLabel("Select models to download")
            self._status_lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
            lay.addWidget(self._status_lbl)

            self._prog = QProgressBar()
            self._prog.setFixedHeight(4)
            self._prog.setTextVisible(False)
            self._prog.setRange(0, 100)
            self._prog.setVisible(False)
            lay.addWidget(self._prog)

            self._log = QTextEdit()
            self._log.setReadOnly(True)
            self._log.setFixedHeight(72)
            self._log.setFont(QFont("Consolas", 10))
            self._log.setVisible(False)
            lay.addWidget(self._log)

            btn_row = QHBoxLayout()
            btn_row.addStretch()
            self._cancel_btn = GlowButton("✕  Cancel", danger=True)
            self._cancel_btn.setVisible(False)
            self._cancel_btn.clicked.connect(self._on_cancel)
            self._close_btn = GlowButton("Close")
            self._close_btn.clicked.connect(self.close)
            self._dl_btn = GlowButton("⬇  DOWNLOAD SELECTED", primary=True)
            self._dl_btn.clicked.connect(self._start_downloads)
            btn_row.addWidget(self._cancel_btn)
            btn_row.addWidget(self._close_btn)
            btn_row.addWidget(self._dl_btn)
            lay.addLayout(btn_row)

        def _start_downloads(self):
            queue = [fn for fn, cb in self._checkboxes.items(
            ) if cb.isChecked() and cb.isEnabled()]
            if not queue:
                self._status_lbl.setText("No models selected.")
                return
            self._queue = queue
            self._current_idx = 0
            self._total = len(queue)
            self._downloading = True
            self._dl_btn.setEnabled(False)
            self._prog.setVisible(True)
            self._prog.setValue(0)
            self._log.setVisible(True)
            self._cancel_btn.setVisible(True)
            self._close_btn.setEnabled(False)
            self._download_next()

        def _download_next(self):
            if self._current_idx >= len(self._queue):
                self._on_all_done()
                return
            fn = self._queue[self._current_idx]
            self._status_lbl.setText(
                f"Downloading {self._current_idx + 1}/{self._total}:  {fn}")
            self._worker = ModelDownloadWorker(fn, self._dest_dir)
            self._worker.log_line.connect(self._on_log)
            self._worker.finished.connect(self._on_file_done)
            self._worker.start()

        def _on_log(self, text: str):
            self._log.append(f'<span style="color:{TEXT_DIM};">{text}</span>')

        def _on_file_done(self, ok: bool, msg: str):
            pct = int((self._current_idx + 1) / self._total * 100)
            self._prog.setValue(pct)
            if ok:
                fn = self._queue[self._current_idx]
                cb = self._checkboxes.get(fn)
                if cb:
                    cb.setChecked(False)
                    cb.setEnabled(False)
            self._current_idx += 1
            self._download_next()

        def _on_all_done(self):
            self._downloading = False
            self._status_lbl.setText(
                f"✓  Download complete  ({self._total} model(s))")
            self._status_lbl.setStyleSheet(f"color:{SUCCESS};font-size:12px;")
            self._prog.setValue(100)
            self._cancel_btn.setVisible(False)
            self._close_btn.setEnabled(True)
            self._dl_btn.setEnabled(True)
            if hasattr(self.models_page, "refresh_list"):
                self.models_page.refresh_list()
            if hasattr(self.models_page, "main_window") and hasattr(self.models_page.main_window, "_populate_mdx_combo"):
                self.models_page.main_window._populate_mdx_combo()

        def _on_cancel(self):
            if self._worker and self._worker.isRunning():
                self._worker.cancel()
                self._worker.terminate()
                self._worker.wait()
            self._downloading = False
            self._status_lbl.setText("Download cancelled.")
            self._cancel_btn.setVisible(False)
            self._close_btn.setEnabled(True)
            self._dl_btn.setEnabled(True)

        def closeEvent(self, e):
            self._on_cancel()
            super().closeEvent(e)

    class ModelsPage(QWidget):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.main_window = parent
            self.models_dict = _load_models_json()
            self.build_ui()

        def build_ui(self):
            lay = QVBoxLayout(self)
            lay.setContentsMargins(32, 28, 32, 28)
            lay.setSpacing(16)

            top = QHBoxLayout()
            title = QLabel("MODELS")
            title.setStyleSheet(
                f"color:{CYAN}; font-size:16px; font-weight:bold; letter-spacing:3px;")
            top.addWidget(title)
            top.addStretch()

            self.dl_btn = GlowButton("⬇  DOWNLOAD MODELS", primary=True)
            self.dl_btn.clicked.connect(self.open_download_dialog)
            top.addWidget(self.dl_btn)

            self.fuse_btn = GlowButton("⊕  FUSE MODELS")
            self.fuse_btn.clicked.connect(self.open_fuse_dialog)
            top.addWidget(self.fuse_btn)

            self.hf_btn = GlowButton("🤗  HF IMPORT")
            self.hf_btn.clicked.connect(self.open_hf_dialog)
            top.addWidget(self.hf_btn)

            lay.addLayout(top)

            hint = QLabel(
                "Discovered models in your data folder. Built-in models are always available.")
            hint.setStyleSheet(f"color:{TEXT_MUTED}; font-size:11px;")
            lay.addWidget(hint)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("background:transparent; border:none;")
            content = QWidget()
            self.grid = QVBoxLayout(content)
            self.grid.setContentsMargins(0, 0, 0, 0)
            self.grid.setSpacing(8)
            scroll.setWidget(content)
            lay.addWidget(scroll, stretch=1)

            self.refresh_list()

        def refresh_list(self):
            while self.grid.count():
                item = self.grid.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()

            self.models_dict = _load_models_json()

            for key, info in self.models_dict.items():
                if info.get("builtin"):
                    self.add_model_row(key, info)

            sep = QFrame()
            sep.setFrameShape(QFrame.Shape.HLine)
            sep.setStyleSheet(f"color:{BORDER};")
            self.grid.addWidget(sep)

            for key, info in self.models_dict.items():
                if not info.get("builtin"):
                    self.add_model_row(key, info)

            self.grid.addStretch()

        def add_model_row(self, key, info):
            row = QFrame()
            row.setStyleSheet(
                f"background:{BG2}; border:1px solid {BORDER2}; border-radius:6px;")
            row.setFixedHeight(60)
            rl = QHBoxLayout(row)
            rl.setContentsMargins(16, 0, 16, 0)

            type_icon = "◈" if info["type"] == "base" else "◈◈"
            icon_lbl = QLabel(type_icon)
            icon_lbl.setStyleSheet(f"color:{CYAN_DIM}; font-size:16px;")
            rl.addWidget(icon_lbl)
            rl.addSpacing(12)

            name_lbl = QLabel(info["name"])
            name_lbl.setStyleSheet(
                f"color:{TEXT}; font-size:13px; font-weight:bold;")
            type_lbl = QLabel(info["type"].upper())
            type_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")
            vbox = QVBoxLayout()
            vbox.setSpacing(2)
            vbox.addWidget(name_lbl)
            vbox.addWidget(type_lbl)
            rl.addLayout(vbox, stretch=1)

            if info["type"] == "base" and not info.get("builtin"):
                path_lbl = QLabel(info.get("file", ""))
                path_lbl.setStyleSheet(f"color:{TEXT_MUTED}; font-size:10px;")
                path_lbl.setWordWrap(True)
                rl.addWidget(path_lbl)

            if not info.get("builtin"):
                del_btn = QPushButton("✕")
                del_btn.setFixedSize(24, 24)
                del_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                del_btn.setStyleSheet(f"""
                    QPushButton {{
                        background:transparent; border:1px solid {ERROR_C}; border-radius:12px;
                        color:{ERROR_C}; font-size:12px;
                    }}
                    QPushButton:hover {{ background:{ERROR_C}; color:#000; }}
                """)
                del_btn.clicked.connect(lambda _, k=key: self.delete_model(k))
                rl.addWidget(del_btn)

            self.grid.addWidget(row)

        def delete_model(self, key):
            ret = QMessageBox.question(self, "Delete Model",
                                       f"Are you sure you want to delete model '{key}'?\nThis will also delete the file if it's a base model.",
                                       QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No)
            if ret == QMessageBox.StandardButton.Yes:
                if _delete_model_file(key):
                    self.refresh_list()
                    if hasattr(self.main_window, "_mdx_combo"):
                        self.main_window._populate_mdx_combo()
                else:
                    QMessageBox.warning(
                        self, "Error", "Could not delete model (built-in or missing).")

        def open_download_dialog(self):
            dlg = ModelDownloadDialog(self, self)
            dlg.exec()

        def open_fuse_dialog(self):
            dlg = ModelFuseDialog(self.models_dict, self)
            if dlg.exec() == QDialog.DialogCode.Accepted:
                result = dlg.get_result()
                key = result["key"]
                if not key or not result["name"]:
                    QMessageBox.warning(
                        self, "Invalid", "Please enter a fused model name.")
                    return
                if result["model_a"] == result["model_b"]:
                    QMessageBox.warning(
                        self, "Invalid", "Please select two different models.")
                    return
                _save_fused_model(
                    key, result["name"], result["model_a"], result["model_b"])
                self.refresh_list()
                if hasattr(self.main_window, "_mdx_combo"):
                    self.main_window._populate_mdx_combo()

        def open_hf_dialog(self):
            dlg = HuggingFaceImportDialog(self, self)
            dlg.exec()

    class StemLabWindow(QMainWindow):
        def __init__(self):
            super().__init__()
            self.setWindowTitle("STEMLAB")
            self.setMinimumSize(860, 680)
            self.resize(1020, 780)
            self.backend, self.backend_desc = HardwareDetector.get_best_backend()
            self.onnx_providers = HardwareDetector.get_onnx_providers(
                self.backend)
            self.worker: Optional[SeparationWorker] = None
            self.batch_worker: Optional[BatchWorker] = None
            self._stems:   dict = {}
            self._out_dir: Optional[Path] = None
            self._song_name: str = ""
            self._active_players: list = []
            self._preview_worker: Optional[PreviewWorker] = None
            self._max_use_toml: bool = False
            self._current_stems_record: Optional[dict] = None
            self._is_batch: bool = False
            self._batch_files: list = []
            self._build_ui()
            self.setStyleSheet(STYLE)
            self._load_history()

        def _build_ui(self):
            root = QWidget()
            self.setCentralWidget(root)
            rl = QVBoxLayout(root)
            rl.setContentsMargins(0, 0, 0, 0)
            rl.setSpacing(0)
            rl.addWidget(self._build_header())
            body = QHBoxLayout()
            body.setContentsMargins(0, 0, 0, 0)
            body.setSpacing(0)
            body.addWidget(self._build_sidebar())
            self._stack = QStackedWidget()
            self._stack.addWidget(self._build_config_page())
            self._stack.addWidget(self._build_running_page())
            self._stack.addWidget(self._build_results_overview())
            self._stack.addWidget(ModelsPage(self))
            self._stack.addWidget(self._build_stems_list_page())
            self._stack.addWidget(self._build_stem_player_page())
            body.addWidget(self._stack, stretch=1)
            rl.addLayout(body, stretch=1)

        def _build_header(self) -> QWidget:
            h = QWidget()
            h.setFixedHeight(54)
            h.setStyleSheet(
                f"background:{BG};border-bottom:1px solid {BORDER};")
            lay = QHBoxLayout(h)
            lay.setContentsMargins(24, 0, 24, 0)
            logo = QLabel(
                "STEM<span style='color:{c}'>LAB</span>".format(c=CYAN))
            logo.setTextFormat(Qt.TextFormat.RichText)
            logo.setStyleSheet(
                "font-size:17px;font-weight:bold;letter-spacing:4px;color:#fff;")
            hw = QLabel(f"⬡  {self.backend_desc}")
            hw.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
            onnx = QLabel("  ·  ".join(self.onnx_providers))
            onnx.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
            lay.addWidget(logo)
            lay.addStretch()
            lay.addWidget(hw)
            lay.addSpacing(16)
            lay.addWidget(onnx)
            return h

        def _build_sidebar(self) -> QWidget:
            s = QWidget()
            s.setFixedWidth(190)
            s.setStyleSheet(
                f"background:{BG2};border-right:1px solid {BORDER};")
            lay = QVBoxLayout(s)
            lay.setContentsMargins(0, 24, 0, 24)
            lay.setSpacing(2)
            self._nav_btns = []
            nav_items = [("01", "CONFIGURE", 0),
                         ("02", "PROCESS",   1),
                         ("03", "RESULTS",   2),
                         ("04", "MODELS",    3)]
            for num, label, idx in nav_items:
                btn = QPushButton(f"  {num}  {label}")
                btn.setCheckable(True)
                btn.setFlat(True)
                btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
                btn.setStyleSheet(f"""
                    QPushButton {{
                        background:transparent; border:none; color:{TEXT_MUTED};
                        font-size:11px; font-weight:bold; letter-spacing:2px;
                        padding:12px 16px; text-align:left;
                    }}
                    QPushButton:checked {{ color:{CYAN}; background:{CYAN_GLOW}; border-left:2px solid {CYAN}; }}
                    QPushButton:hover:!checked {{ color:{TEXT_DIM}; }}
                """)
                btn.clicked.connect(lambda _, i=idx: self._nav_to(i))
                self._nav_btns.append(btn)
                lay.addWidget(btn)
            self._nav_btns[0].setChecked(True)
            lay.addStretch()
            ver = QLabel("v1.1 Beta  ·  StemLab")
            ver.setStyleSheet(
                f"color:{TEXT_MUTED};font-size:10px;padding:0 16px;")
            lay.addWidget(ver)
            return s

        def _nav_to(self, idx: int):
            for i, b in enumerate(self._nav_btns):
                b.setChecked(i == min(idx, len(self._nav_btns)-1))
            self._stack.setCurrentIndex(idx)

        def _populate_mdx_combo(self):
            if not hasattr(self, "_mdx_combo"):
                return
            self._mdx_combo.clear()
            models = _load_models_json()
            for key, info in models.items():
                if (info["type"] in ("base", "fused") or key == "PHANTOM") and _is_vocal_model(key, info):
                    self._mdx_combo.addItem(info["name"], key)

        def _build_config_page(self) -> QWidget:
            inner = QWidget()
            lay = QVBoxLayout(inner)
            lay.setContentsMargins(32, 28, 32, 28)
            lay.setSpacing(18)

            lay.addWidget(_sec_lbl("SEPARATION QUALITY"))
            self._preset_cards: dict[str, PresetCard] = {}
            grid = QHBoxLayout()
            grid.setSpacing(8)
            for key in ["fast", "fine-tuned", "professional", "max"]:
                card = PresetCard(key)
                card.clicked.connect(self._on_preset_click)
                self._preset_cards[key] = card
                grid.addWidget(card)
            lay.addLayout(grid)

            stem_frame = QFrame()
            stem_frame.setStyleSheet(
                f"background:{BG2};border:1px solid {BORDER};border-radius:8px;")
            sf_lay = QHBoxLayout(stem_frame)
            sf_lay.setContentsMargins(20, 14, 20, 14)
            sf_lay.setSpacing(16)
            self._seven_stem_chk = QCheckBox("7-Stem Separation")
            self._seven_stem_chk.setStyleSheet(
                f"color:{TEXT};font-size:13px;font-weight:bold;")
            seven_desc = QLabel(
                "Uses htdemucs_6s  ·  Vocals / Bass / Drums / Guitar / Piano / Other / Instrumental")
            seven_desc.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
            sf_lay.addWidget(self._seven_stem_chk)
            sf_lay.addSpacing(8)
            sf_lay.addWidget(seven_desc, stretch=1)
            lay.addWidget(stem_frame)

            self._max_panel = QFrame()
            self._max_panel.setStyleSheet(
                f"background:{BG2};border:1px solid {BORDER};border-radius:8px;")
            mp = QVBoxLayout(self._max_panel)
            mp.setContentsMargins(20, 16, 20, 16)
            mp.setSpacing(14)
            mp_hdr_row = QHBoxLayout()
            mp_hdr_row.addWidget(_sec_lbl("MAX MODE OPTIONS"))
            mp_hdr_row.addStretch()
            toml_available = _TOML_OK and CONFIG_TOML.exists()
            self._toml_badge = QLabel(
                "config.toml ✓" if toml_available else
                ("config.toml — not found" if _TOML_OK else "config.toml — tomllib unavailable")
            )
            self._toml_badge.setStyleSheet(
                f"color:{'#00cc66' if toml_available else TEXT_MUTED};font-size:10px;")
            mp_hdr_row.addWidget(self._toml_badge)
            mp.addLayout(mp_hdr_row)

            self._toml_toggle_btn = QPushButton("⊡  Use config.toml")
            self._toml_toggle_btn.setCheckable(True)
            self._toml_toggle_btn.setEnabled(toml_available)
            self._toml_toggle_btn.setCursor(
                QCursor(Qt.CursorShape.PointingHandCursor))
            self._toml_toggle_btn.setFixedHeight(32)
            self._toml_toggle_btn.setStyleSheet(f"""
                QPushButton {{
                    background:transparent; border:1px solid {BORDER2}; border-radius:6px;
                    color:{TEXT_MUTED}; font-size:11px; padding:4px 14px; text-align:left;
                }}
                QPushButton:checked {{ background:{CYAN_DARK}; border:1px solid {CYAN}; color:{CYAN}; }}
                QPushButton:hover:!checked {{ border-color:{CYAN_DIM}; color:{TEXT_DIM}; }}
                QPushButton:disabled {{ color:{TEXT_MUTED}; border-color:{BORDER}; opacity:0.5; }}
            """)
            self._toml_toggle_btn.toggled.connect(self._on_toml_toggle)

            self._toml_create_btn = QPushButton("+ Create config.toml")
            self._toml_create_btn.setCursor(
                QCursor(Qt.CursorShape.PointingHandCursor))
            self._toml_create_btn.setFixedHeight(32)
            self._toml_create_btn.setVisible(not toml_available and _TOML_OK)
            self._toml_create_btn.setStyleSheet(f"""
                QPushButton {{
                    background:transparent; border:1px solid {BORDER2}; border-radius:6px;
                    color:{TEXT_MUTED}; font-size:11px; padding:4px 14px;
                }}
                QPushButton:hover {{ border-color:{CYAN_DIM}; color:{TEXT}; }}
            """)
            self._toml_create_btn.clicked.connect(self._create_config_toml)

            toml_row = QHBoxLayout()
            toml_row.addWidget(self._toml_toggle_btn)
            toml_row.addWidget(self._toml_create_btn)
            toml_row.addStretch()
            mp.addLayout(toml_row)

            self._toml_active_lbl = QLabel(
                f"⊙  Max options will be loaded from  {CONFIG_TOML.name}  —  manual controls are locked")
            self._toml_active_lbl.setStyleSheet(
                f"color:{CYAN_DIM};font-size:11px;padding:2px 0;")
            self._toml_active_lbl.setVisible(False)
            mp.addWidget(self._toml_active_lbl)

            div0 = QFrame()
            div0.setFrameShape(QFrame.Shape.HLine)
            div0.setStyleSheet(f"color:{BORDER};")
            mp.addWidget(div0)

            def _opt_row(label_text, widget):
                r = QHBoxLayout()
                r.setSpacing(12)
                lbl = QLabel(label_text)
                lbl.setStyleSheet(f"color:{TEXT_DIM};font-size:13px;")
                lbl.setFixedWidth(120)
                r.addWidget(lbl)
                r.addWidget(widget, stretch=1)
                return r

            self._mdx_combo = QComboBox()
            self._populate_mdx_combo()
            self._mdx_combo.setCurrentIndex(0)
            mp.addLayout(_opt_row("Vocal model", self._mdx_combo))

            self._dr_combo = QComboBox()
            for k, v in DEREVERB_MODELS.items():
                self._dr_combo.addItem(v["display"], k)
            self._dr_combo.setCurrentIndex(
                list(DEREVERB_MODELS.keys()).index("both"))
            mp.addLayout(_opt_row("Post-process", self._dr_combo))

            div = QFrame()
            div.setFrameShape(QFrame.Shape.HLine)
            div.setStyleSheet(f"color:{BORDER};")
            mp.addWidget(div)
            self._debleed_chk = QCheckBox(
                "De-bleed  —  attenuate rhythmic bleed (snare / hi-hat) from instrumental")
            self._debleed_chk.setStyleSheet(
                f"color:{TEXT_DIM};font-size:12px;")
            mp.addWidget(self._debleed_chk)

            self._max_manual_widgets = [
                self._mdx_combo, self._dr_combo, self._debleed_chk]
            self._max_panel.setVisible(False)
            lay.addWidget(self._max_panel)

            lay.addWidget(_sec_lbl("EXPORT FORMAT"))
            fmt_frame = QFrame()
            fmt_frame.setStyleSheet(
                f"background:{BG2};border:1px solid {BORDER};border-radius:8px;")
            ff = QHBoxLayout(fmt_frame)
            ff.setContentsMargins(20, 14, 20, 14)
            ff.setSpacing(28)
            self._fmt_wav = QRadioButton("WAV  (lossless · 24-bit)")
            self._fmt_mp3 = QRadioButton("MP3  (320 kbps)")
            self._fmt_flac = QRadioButton("FLAC  (lossless compressed)")
            self._fmt_wav.setChecked(True)
            for rb in (self._fmt_wav, self._fmt_mp3, self._fmt_flac):
                rb.setStyleSheet(f"color:{TEXT};font-size:13px;")
                ff.addWidget(rb)
            ff.addStretch()
            lay.addWidget(fmt_frame)

            lay.addWidget(_sec_lbl("INPUT / OUTPUT"))
            io = QFrame()
            io.setStyleSheet(
                f"background:{BG2};border:1px solid {BORDER};border-radius:8px;")
            il = QVBoxLayout(io)
            il.setContentsMargins(20, 16, 20, 16)
            il.setSpacing(10)

            self._batch_mode_chk = QCheckBox("Batch Mode")
            self._batch_mode_chk.toggled.connect(self._toggle_batch_mode)
            il.addWidget(self._batch_mode_chk)

            self._input_stack = QStackedWidget()
            single_page = QWidget()
            single_lay = QHBoxLayout(single_page)
            single_lay.setContentsMargins(0, 0, 0, 0)
            self._audio_edit = QLineEdit()
            self._audio_edit.setPlaceholderText(
                "Drag & drop or browse for audio file...")
            self._audio_edit.setReadOnly(False)
            self._audio_edit.textChanged.connect(self._update_start_btn)
            self._in_browse_btn = GlowButton("Browse")
            self._in_browse_btn.clicked.connect(self._browse_audio_single)
            single_lay.addWidget(self._audio_edit, stretch=1)
            single_lay.addWidget(self._in_browse_btn)
            self._input_stack.addWidget(single_page)

            batch_page = QWidget()
            batch_lay = QVBoxLayout(batch_page)
            batch_lay.setContentsMargins(0, 0, 0, 0)
            self._batch_list = QListWidget()
            self._batch_list.setSelectionMode(
                QAbstractItemView.SelectionMode.ExtendedSelection)
            self._batch_list.setStyleSheet(
                f"background:{BG3};border:1px solid {BORDER2};border-radius:4px;")
            batch_btn_row = QHBoxLayout()
            self._batch_add_btn = GlowButton("Add Files")
            self._batch_add_btn.clicked.connect(self._add_batch_files)
            self._batch_remove_btn = GlowButton("Remove Selected")
            self._batch_remove_btn.clicked.connect(self._remove_batch_files)
            batch_btn_row.addWidget(self._batch_add_btn)
            batch_btn_row.addWidget(self._batch_remove_btn)
            batch_btn_row.addStretch()
            batch_lay.addWidget(self._batch_list)
            batch_lay.addLayout(batch_btn_row)
            self._input_stack.addWidget(batch_page)

            il.addWidget(self._input_stack)

            out_row = QHBoxLayout()
            out_lbl = QLabel("Output")
            out_lbl.setStyleSheet(f"color:{TEXT_MUTED};font-size:12px;")
            out_lbl.setFixedWidth(56)
            self._out_edit = QLineEdit()
            self._out_edit.setPlaceholderText(
                "Output directory (auto-filled from input)")
            self._out_btn = GlowButton("Browse")
            self._out_btn.clicked.connect(self._browse_output)
            out_row.addWidget(out_lbl)
            out_row.addWidget(self._out_edit, stretch=1)
            out_row.addWidget(self._out_btn)
            il.addLayout(out_row)

            self._batch_info_lbl = QLabel("")
            self._batch_info_lbl.setStyleSheet(
                f"color:{CYAN_DIM};font-size:11px;")
            self._batch_info_lbl.setVisible(False)
            il.addWidget(self._batch_info_lbl)

            lay.addWidget(io)
            lay.addStretch()

            br = QHBoxLayout()
            br.setSpacing(10)
            br.addStretch()
            self._preview_btn = GlowButton("⬡  PREVIEW  (10s)", primary=False)
            self._preview_btn.setFixedHeight(44)
            self._preview_btn.setEnabled(False)
            self._preview_btn.setToolTip(
                "Run a quick 10-second separation preview")
            self._preview_btn.setStyleSheet(f"""
                QPushButton {{
                    background:transparent; border:1px solid {BORDER2}; border-radius:6px;
                    color:{TEXT_DIM}; font-size:12px; padding:9px 22px; letter-spacing:1px;
                }}
                QPushButton:hover   {{ border-color:{CYAN_DIM}; color:{CYAN}; background:{CYAN_GLOW}; }}
                QPushButton:pressed {{ background:{CYAN_DARK}; }}
                QPushButton:disabled {{ border-color:{BORDER}; color:{TEXT_MUTED}; }}
            """)
            self._preview_btn.clicked.connect(self._open_preview)
            self._start_btn = GlowButton("▶  START SEPARATION", primary=True)
            self._start_btn.setFixedHeight(44)
            self._start_btn.setEnabled(False)
            self._start_btn.clicked.connect(self._start)
            br.addWidget(self._preview_btn)
            br.addWidget(self._start_btn)
            lay.addLayout(br)

            self._selected_preset = "fine-tuned"
            self._preset_cards["fine-tuned"].set_active(True)

            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setWidget(inner)
            scroll.setStyleSheet("background:transparent;border:none;")
            outer = QWidget()
            ol = QVBoxLayout(outer)
            ol.setContentsMargins(0, 0, 0, 0)
            ol.setSpacing(0)
            ol.addWidget(scroll)
            return outer

        def _toggle_batch_mode(self, checked: bool):
            self._input_stack.setCurrentIndex(1 if checked else 0)
            self._is_batch = checked
            self._out_edit.setEnabled(not checked)
            self._out_btn.setEnabled(not checked)
            if checked:
                self._out_edit.setText(str(Path.cwd() / "Stem_batch"))
                self._batch_info_lbl.setVisible(True)
            else:
                self._batch_info_lbl.setVisible(False)
                self._batch_list.clear()
                self._batch_files = []
            self._update_start_btn()

        def _add_batch_files(self):
            paths, _ = QFileDialog.getOpenFileNames(
                self, "Add Audio File(s) to Batch", "", AUDIO_FILTER)
            for p in paths:
                if p not in self._batch_files:
                    self._batch_files.append(p)
                    self._batch_list.addItem(Path(p).name)
            self._update_batch_info()
            self._update_start_btn()

        def _remove_batch_files(self):
            selected = self._batch_list.selectedItems()
            for item in selected:
                row = self._batch_list.row(item)
                self._batch_list.takeItem(row)
                del self._batch_files[row]
            self._update_batch_info()
            self._update_start_btn()

        def _update_batch_info(self):
            n = len(self._batch_files)
            if n == 0:
                self._batch_info_lbl.setText("")
                return
            names = "  ·  ".join(Path(p).name for p in self._batch_files[:3])
            if n > 3:
                names += f"  ...+{n-3} more"
            self._batch_info_lbl.setText(f"⊞  Batch: {names}")

        def _browse_audio_single(self):
            path, _ = QFileDialog.getOpenFileName(
                self, "Select Audio File", "", AUDIO_FILTER)
            if path:
                self._audio_edit.setText(path)
                if not self._out_edit.text():
                    self._out_edit.setText(
                        str(Path(path).parent / Path(path).stem))

        def _browse_output(self):
            p = QFileDialog.getExistingDirectory(
                self, "Select Output Directory")
            if p:
                self._out_edit.setText(p)

        def _update_start_btn(self):
            if self._is_batch:
                enabled = len(self._batch_files) > 0
            else:
                enabled = bool(self._audio_edit.text().strip())
            self._start_btn.setEnabled(enabled)
            self._preview_btn.setEnabled(enabled and not self._is_batch)

        def dragEnterEvent(self, e):
            if e.mimeData().hasUrls():
                e.acceptProposedAction()

        def dropEvent(self, e):
            urls = e.mimeData().urls()
            paths = [u.toLocalFile() for u in urls
                     if Path(u.toLocalFile()).suffix.lower() in AUDIO_EXTS]
            if not paths:
                return
            if self._is_batch:
                for p in paths:
                    if p not in self._batch_files:
                        self._batch_files.append(p)
                        self._batch_list.addItem(Path(p).name)
                self._update_batch_info()
                self._update_start_btn()
            else:
                self._audio_edit.setText(paths[0])
                if not self._out_edit.text():
                    self._out_edit.setText(
                        str(Path(paths[0]).parent / Path(paths[0]).stem))

        def _on_preset_click(self, key: str):
            self._selected_preset = key
            for k, c in self._preset_cards.items():
                c.set_active(k == key)
            self._max_panel.setVisible(key == "max")

        def _on_toml_toggle(self, checked: bool):
            self._max_use_toml = checked
            self._toml_active_lbl.setVisible(checked)
            for w in self._max_manual_widgets:
                w.setEnabled(not checked)

        def _create_config_toml(self):
            try:
                CONFIG_TOML.write_text(
                    _example_config_toml_text(), encoding="utf-8")
                self._toml_create_btn.setVisible(False)
                self._toml_badge.setText("config.toml ✓")
                self._toml_badge.setStyleSheet("color:#00cc66;font-size:10px;")
                self._toml_toggle_btn.setEnabled(True)
                QMessageBox.information(
                    self, "config.toml created",
                    f"Example config.toml written to:\n{CONFIG_TOML}\n\nEdit it to customise settings.")
            except Exception as e:
                QMessageBox.warning(
                    self, "Could not create config.toml", str(e))

        def _load_history(self):
            records = _songs_json_validate_and_prune()
            if not records:
                return
            for r in reversed(records):
                self._add_song_card_from_record(r)
            latest = records[-1]
            self._populate_stems_list_from_record(latest)

        def _open_preview(self):
            ap = Path(self._audio_edit.text().strip())
            if not ap.exists():
                QMessageBox.warning(self, "Not Found",
                                    f"File not found:\n{ap}")
                return
            preset = self._selected_preset
            fmt = "mp3" if self._fmt_mp3.isChecked() else "wav"
            if self._fmt_flac.isChecked():
                fmt = "flac"
            cfg = self._build_cfg_for_start(preset, fmt)
            dlg = PreviewDialog(cfg, ap, self)
            dlg.exec()

        def _build_cfg_for_start(self, preset: str, fmt: str) -> SeparationConfig:
            seven = self._seven_stem_chk.isChecked()
            max_opts = None
            toml_ov = None
            if preset == "max" and not seven:
                if self._max_use_toml:
                    toml_ov = _load_toml_overrides("max")
                else:
                    dr_key = self._dr_combo.currentData()
                    max_opts = MaxModeOptions(
                        mdx_model_key=self._mdx_combo.currentData(),
                        dereverb_model_key=None if dr_key == "none" else dr_key,
                        debleed=self._debleed_chk.isChecked(),
                    )
            else:
                toml_ov = _load_toml_overrides(preset)
            return SeparationConfig.from_preset(
                preset, self.backend, max_opts,
                export_format=fmt, toml_overrides=toml_ov,
                seven_stem=seven)

        def _start(self):
            preset = self._selected_preset
            fmt = "mp3" if self._fmt_mp3.isChecked() else "wav"
            if self._fmt_flac.isChecked():
                fmt = "flac"
            cfg = self._build_cfg_for_start(preset, fmt)
            self._log.clear()
            self._progress.setValue(0)
            self._pulse.start()
            self._nav_to(1)

            if self._is_batch:
                self._start_batch(cfg)
            else:
                self._start_single(cfg)

        def _start_single(self, cfg: SeparationConfig):
            ap = Path(self._audio_edit.text().strip())
            ot = self._out_edit.text().strip()
            od = Path(ot) if ot else (ap.parent / ap.stem)
            if not ap.exists():
                QMessageBox.warning(self, "Not Found",
                                    f"File not found:\n{ap}")
                return
            self._status_lbl.setText("Starting...")
            self._out_dir = od
            self._stems = {}
            self._song_name = ap.stem
            self._current_cfg = cfg
            self.worker = SeparationWorker(cfg, ap, od)
            self.worker.log_line.connect(self._on_log)
            self.worker.progress.connect(self._progress.setValue)
            self.worker.stem_ready.connect(
                lambda n, p: self._stems.update({n: p}))
            self.worker.finished.connect(self._on_finished)
            self.worker.start()

        def _start_batch(self, cfg: SeparationConfig):
            batch_dir = Path(self._out_edit.text().strip(
            )) if self._out_edit.text() else Path.cwd() / "Stem_batch"
            batch_dir.mkdir(parents=True, exist_ok=True)
            self._status_lbl.setText(
                f"Batch: 0/{len(self._batch_files)} files...")
            self._current_cfg = cfg
            self._out_dir = batch_dir
            self.batch_worker = BatchWorker(cfg, self._batch_files, batch_dir)
            self.batch_worker.log_line.connect(self._on_log)
            self.batch_worker.progress.connect(self._progress.setValue)
            self.batch_worker.file_started.connect(self._on_batch_file_started)
            self.batch_worker.file_done.connect(self._on_batch_file_done)
            self.batch_worker.finished.connect(self._on_batch_finished)
            self.batch_worker.start()

        def _on_batch_file_started(self, fname: str, idx: int, total: int):
            self._status_lbl.setText(f"Batch [{idx}/{total}]  {fname}")

        def _on_batch_file_done(self, fname: str, ok: bool):
            icon = "✓" if ok else "✗"
            self._on_log(f"  {icon}  {fname}")

        def _on_batch_finished(self, ok: bool, msg: str):
            self._pulse.stop()
            if ok:
                self._status_lbl.setText(f"✓  {msg}")
            else:
                self._status_lbl.setText(f"✗  {msg}")
            self._build_stem_grid({}, self._out_dir, "Batch")
            QTimer.singleShot(500, lambda: self._nav_to(2))

        def _build_running_page(self) -> QWidget:
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(32, 28, 32, 28)
            lay.setSpacing(14)
            sr = QHBoxLayout()
            self._pulse = PulsingDot()
            self._status_lbl = QLabel("Initialising...")
            self._status_lbl.setStyleSheet(
                f"color:{CYAN};font-size:13px;font-weight:bold;")
            self._cancel_btn = GlowButton("Cancel", danger=True)
            self._cancel_btn.clicked.connect(self._cancel)
            sr.addWidget(self._pulse)
            sr.addSpacing(8)
            sr.addWidget(self._status_lbl)
            sr.addStretch()
            sr.addWidget(self._cancel_btn)
            lay.addLayout(sr)
            self._progress = QProgressBar()
            self._progress.setFixedHeight(5)
            self._progress.setTextVisible(False)
            self._progress.setRange(0, 100)
            lay.addWidget(self._progress)
            lay.addSpacing(4)
            lay.addWidget(_sec_lbl("PROCESS LOG"))
            self._log = QTextEdit()
            self._log.setReadOnly(True)
            self._log.setFont(QFont("Consolas", 11))
            lay.addWidget(self._log, stretch=1)
            return w

        def _on_log(self, text: str):
            self._log.append(f'<span style="color:{TEXT_DIM};">{text}</span>')
            self._status_lbl.setText(text[:90])

        def _on_finished(self, ok: bool, msg: str):
            self._pulse.stop()
            if ok:
                self._status_lbl.setText("✓  Complete")
                cfg = getattr(self, "_current_cfg", None)
                _songs_json_append(
                    self._song_name, self._out_dir, self._stems,
                    preset=cfg.quality_preset if cfg else "?",
                    model=cfg.model if cfg else "unknown",
                    backend_desc=self.backend_desc,
                )
                self._build_stem_grid(
                    self._stems, self._out_dir, self._song_name)
                QTimer.singleShot(500, lambda: self._nav_to(2))
            else:
                self._status_lbl.setText(f"✗  {msg}")
                self._log.append(
                    f'<span style="color:{ERROR_C};">FAILED: {msg}</span>')

        def _cancel(self):
            if self.batch_worker and self.batch_worker.isRunning():
                self.batch_worker.cancel()
                self.batch_worker.terminate()
                self.batch_worker.wait()
            if self.worker and self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait()
            self._pulse.stop()
            self._nav_to(0)

        def _build_results_overview(self) -> QWidget:
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(32, 28, 32, 28)
            lay.setSpacing(16)
            top = QHBoxLayout()
            hdr = QLabel("RESULTS")
            hdr.setStyleSheet(
                f"color:{CYAN};font-size:16px;font-weight:bold;letter-spacing:3px;")
            new_btn = GlowButton("◀  New Separation", primary=True)
            new_btn.clicked.connect(self._new_separation)
            top.addWidget(hdr)
            top.addStretch()
            top.addWidget(new_btn)
            lay.addLayout(top)
            div = QFrame()
            div.setFrameShape(QFrame.Shape.HLine)
            div.setStyleSheet(f"color:{BORDER};")
            lay.addWidget(div)
            hint = QLabel(
                "Click a song card to browse its stems")
            hint.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
            lay.addWidget(hint)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("background:transparent;border:none;")
            self._song_card_area = QWidget()
            self._song_card_area.setStyleSheet("background:transparent;")
            self._song_card_lay = QVBoxLayout(self._song_card_area)
            self._song_card_lay.setContentsMargins(0, 8, 0, 8)
            self._song_card_lay.setSpacing(10)
            self._song_card_lay.addStretch()
            scroll.setWidget(self._song_card_area)
            lay.addWidget(scroll, stretch=1)
            bot = QHBoxLayout()
            bot.addStretch()
            open_btn = GlowButton("⊞  Open Output Folder")
            open_btn.clicked.connect(self._open_folder)
            bot.addWidget(open_btn)
            lay.addLayout(bot)
            return w

        def _add_song_card_from_record(self, r: dict):
            out_dir = r.get("out_dir", "")
            song_name = r.get("song_name", Path(out_dir).name)
            preset = r.get("preset", "")
            date = r.get("date", "")
            n_stems = len(r.get("stems", {}))
            sc = SongCard(song_name, out_dir, n_stems,
                          preset=preset, date=date, record=r)
            sc.clicked.connect(self._on_song_card_click)
            self._song_card_lay.insertWidget(
                self._song_card_lay.count() - 1, sc)

        def _on_song_card_click(self, record: dict):
            self._populate_stems_list_from_record(record)
            self._nav_to(4)

        def _populate_stems_list_from_record(self, record: dict):
            stems = record.get("stems", {})
            out_dir = record.get("out_dir", "")
            song_name = record.get("song_name", Path(out_dir).name)
            self._song_name = song_name
            self._out_dir = Path(out_dir)
            self._stems = stems
            self._current_stems_record = record
            while self._stems_list_lay.count() > 1:
                item = self._stems_list_lay.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self._stems_list_title.setText(song_name.upper())
            self._stems_list_path.setText(out_dir)
            stem_order = ["vocals", "instrumental", "bass",
                          "drums", "guitar", "piano", "other"]
            for name in stem_order:
                path = stems.get(name)
                if path and Path(path).exists():
                    row = StemRowCard(name, str(path))
                    row.clicked.connect(self._open_player)
                    self._stems_list_lay.insertWidget(
                        self._stems_list_lay.count() - 1, row)

        def _build_stems_list_page(self) -> QWidget:
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(32, 28, 32, 28)
            lay.setSpacing(16)
            top = QHBoxLayout()
            back_btn = GlowButton("‹  Results")
            back_btn.clicked.connect(lambda: self._nav_to(2))
            crumb = QHBoxLayout()
            crumb.setSpacing(6)
            crumb_sep = QLabel("›")
            crumb_sep.setStyleSheet(f"color:{TEXT_MUTED};font-size:14px;")
            self._stems_list_title = QLabel()
            self._stems_list_title.setStyleSheet(
                f"color:{CYAN};font-size:15px;font-weight:bold;letter-spacing:3px;")
            crumb.addWidget(crumb_sep)
            crumb.addWidget(self._stems_list_title)
            crumb.addStretch()
            self._stems_list_path = QLabel()
            self._stems_list_path.setStyleSheet(
                f"color:{TEXT_MUTED};font-size:11px;")
            top.addWidget(back_btn)
            top.addSpacing(8)
            top.addLayout(crumb, stretch=1)
            lay.addLayout(top)
            self._stems_list_path_row = QHBoxLayout()
            path_icon = QLabel("⊙")
            path_icon.setStyleSheet(f"color:{TEXT_MUTED};font-size:12px;")
            self._stems_list_path_row.addWidget(path_icon)
            self._stems_list_path_row.addWidget(self._stems_list_path)
            self._stems_list_path_row.addStretch()
            lay.addLayout(self._stems_list_path_row)
            div = QFrame()
            div.setFrameShape(QFrame.Shape.HLine)
            div.setStyleSheet(f"color:{BORDER};")
            lay.addWidget(div)
            hint = QLabel("Click a stem to open the waveform player")
            hint.setStyleSheet(
                f"color:{TEXT_MUTED};font-size:11px;letter-spacing:1px;")
            lay.addWidget(hint)
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setStyleSheet("background:transparent;border:none;")
            self._stems_list_w = QWidget()
            self._stems_list_w.setStyleSheet("background:transparent;")
            self._stems_list_lay = QVBoxLayout(self._stems_list_w)
            self._stems_list_lay.setContentsMargins(0, 4, 0, 8)
            self._stems_list_lay.setSpacing(8)
            self._stems_list_lay.addStretch()
            scroll.setWidget(self._stems_list_w)
            lay.addWidget(scroll, stretch=1)
            return w

        def _build_stem_player_page(self) -> QWidget:
            w = QWidget()
            lay = QVBoxLayout(w)
            lay.setContentsMargins(32, 28, 32, 28)
            lay.setSpacing(16)
            top = QHBoxLayout()
            back_btn = GlowButton("‹  Stems")
            back_btn.clicked.connect(
                lambda: (self._stop_players(), self._nav_to(4)))
            crumb = QHBoxLayout()
            crumb.setSpacing(6)
            crumb_sep = QLabel("›")
            crumb_sep.setStyleSheet(f"color:{TEXT_MUTED};font-size:14px;")
            self._player_page_title = QLabel()
            self._player_page_title.setStyleSheet(
                f"color:{CYAN};font-size:15px;font-weight:bold;letter-spacing:2px;")
            self._player_page_song = QLabel()
            self._player_page_song.setStyleSheet(
                f"color:{TEXT_MUTED};font-size:12px;")
            crumb.addWidget(crumb_sep)
            crumb.addWidget(self._player_page_title)
            crumb.addSpacing(8)
            crumb.addWidget(self._player_page_song, stretch=1)
            top.addWidget(back_btn)
            top.addSpacing(8)
            top.addLayout(crumb, stretch=1)
            lay.addLayout(top)
            div = QFrame()
            div.setFrameShape(QFrame.Shape.HLine)
            div.setStyleSheet(f"color:{BORDER};")
            lay.addWidget(div)
            self._player_container = QWidget()
            self._player_container.setStyleSheet("background:transparent;")
            self._player_lay = QVBoxLayout(self._player_container)
            self._player_lay.setContentsMargins(0, 0, 0, 0)
            self._player_lay.setSpacing(0)
            lay.addWidget(self._player_container, stretch=1)
            return w

        def _build_stem_grid(self, stems: dict, out_dir: Path, song_name: str):
            records = _songs_json_validate_and_prune()
            while self._song_card_lay.count() > 1:
                item = self._song_card_lay.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            for r in reversed(records):
                self._add_song_card_from_record(r)
            latest = next(
                (r for r in reversed(records)
                 if r.get("out_dir") == str(out_dir.resolve())),
                None,
            )
            if latest:
                self._populate_stems_list_from_record(latest)
            else:
                self._stems_list_title.setText(song_name.upper())
                self._stems_list_path.setText(str(out_dir))
                while self._stems_list_lay.count() > 1:
                    item = self._stems_list_lay.takeAt(0)
                    if item.widget():
                        item.widget().deleteLater()
                stem_order = ["vocals", "instrumental", "bass",
                              "drums", "guitar", "piano", "other"]
                for name in stem_order:
                    path = stems.get(name)
                    if path and Path(path).exists():
                        row = StemRowCard(name, str(path))
                        row.clicked.connect(self._open_player)
                        self._stems_list_lay.insertWidget(
                            self._stems_list_lay.count() - 1, row)

        def _open_player(self, stem_name: str, file_path: str):
            self._stop_players()
            while self._player_lay.count():
                item = self._player_lay.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self._player_page_title.setText(stem_name.upper())
            self._player_page_song.setText(self._song_name)
            player = StemPlayer(stem_name, file_path)
            self._active_players = [player]
            self._player_lay.addWidget(player)
            self._player_lay.addStretch()
            self._nav_to(5)

        def _stop_players(self):
            for p in self._active_players:
                try:
                    p.stop_playback()
                except Exception:
                    pass
            self._active_players = []

        def _new_separation(self):
            self._stop_players()
            self._is_batch = False
            self._batch_files = []
            self._batch_mode_chk.setChecked(False)
            self._audio_edit.clear()
            self._out_edit.clear()
            self._batch_list.clear()
            self._nav_to(0)

        def _open_folder(self):
            p = self._out_dir
            if not p or not Path(p).exists():
                QMessageBox.warning(self, "Folder Not Found",
                                    f"Output folder does not exist:\n{p}")
                return
            try:
                if sys.platform == "win32":
                    os.startfile(str(p))
                elif sys.platform == "darwin":
                    subprocess.Popen(["open", str(p)])
                else:
                    subprocess.Popen(["xdg-open", str(p)])
            except Exception as e:
                QMessageBox.warning(self, "Cannot Open Folder", str(e))

        def closeEvent(self, e):
            self._stop_players()
            if self.batch_worker and self.batch_worker.isRunning():
                self.batch_worker.cancel()
                self.batch_worker.terminate()
                self.batch_worker.wait()
            if self.worker and self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait()
            super().closeEvent(e)

    app = QApplication(sys.argv + ["-style", "Fusion"])
    app.setApplicationName("StemLab")
    pal = QPalette()
    for role, col in [
        (QPalette.ColorRole.Window,          BG),
        (QPalette.ColorRole.WindowText,      TEXT),
        (QPalette.ColorRole.Base,            BG2),
        (QPalette.ColorRole.AlternateBase,   BG3),
        (QPalette.ColorRole.Text,            TEXT),
        (QPalette.ColorRole.Button,          BG2),
        (QPalette.ColorRole.ButtonText,      TEXT),
        (QPalette.ColorRole.Highlight,       CYAN_DARK),
        (QPalette.ColorRole.HighlightedText, CYAN),
        (QPalette.ColorRole.ToolTipBase,     BG3),
        (QPalette.ColorRole.ToolTipText,     TEXT),
    ]:
        pal.setColor(role, QColor(col))
    app.setPalette(pal)
    win = StemLabWindow()
    win.setAcceptDrops(True)
    win.show()
    sys.exit(app.exec())


_TC = "\033[96m"
_GRAY = "\033[90m"
_WHT = "\033[97m"
_DIM = "\033[2m"
_RST = "\033[0m"
_W = 62
_DIV = f"{_TC}{'─' * _W}{_RST}"


def _tui_info(text): print(f"    {_GRAY}[i]{_RST} {text}")
def _tui_warn(text): print(f"    {_WHT}[⚠]{_RST}  {text}")
def _tui_done(text): print(f"    {_TC}[✓]{_RST}  {text}")


def _tui_opt(number, label):
    print(f"  {_TC}[{_GRAY}{number}{_TC}]{_RST}  {_GRAY}{label}{_RST}")


def _tui_prompt_max_options() -> MaxModeOptions:
    print(f"\n{_DIV}\n  MAX MODE\n{_DIV}")
    print(f"\n  {_GRAY}Vocal separation model:{_RST}\n")
    models = _load_models_json()
    model_list = []
    for key, info in models.items():
        if (info["type"] in ("base", "fused") or key == "PHANTOM") and _is_vocal_model(key, info):
            model_list.append((key, info["name"]))
    for i, (key, name) in enumerate(model_list, 1):
        suffix = "  ← recommended" if key == "PHANTOM" else ""
        print(f"  {_TC}[{_GRAY}{i}{_TC}]{_RST}  {_GRAY}{name}{suffix}{_RST}")
    while True:
        raw = input(f"\n  {_TC}>{_RST} ").strip() or "1"
        if raw.isdigit() and 1 <= int(raw) <= len(model_list):
            mdx_key = model_list[int(raw) - 1][0]
            break
        print(f"  {_GRAY}Invalid choice.{_RST}")
    _tui_done(models[mdx_key]["name"] if mdx_key in models else "PHANTOM")

    print(f"\n{_DIV}\n  {_GRAY}Post-processing{_RST}\n")
    pp_keys = list(DEREVERB_MODELS.keys())
    both_idx = pp_keys.index("both")
    for i, key in enumerate(pp_keys):
        suffix = "  ← recommended" if key == "both" else ""
        print(
            f"  {_TC}[{_GRAY}{i}{_TC}]{_RST}  {_GRAY}{DEREVERB_MODELS[key]['display']}{suffix}{_RST}")
    while True:
        raw = input(f"\n  {_TC}>{_RST} ").strip() or str(both_idx)
        if raw.isdigit() and 0 <= int(raw) < len(pp_keys):
            dr_key = pp_keys[int(raw)]
            break
        print(f"  {_GRAY}Invalid choice.{_RST}")
    dr_key = None if dr_key == "none" else dr_key
    _tui_done(DEREVERB_MODELS[dr_key]["display"]
              if dr_key else "Post-processing disabled")

    print(f"\n{_DIV}\n  {_GRAY}De-bleed{_RST}\n")
    print(
        f"  {_DIM}Attenuates rhythmic bleed (snare/hi-hat) that leaked through.{_RST}\n")
    debleed = input(
        f"  Enable de-bleed? {_GRAY}[y/N]{_RST}: ").strip().lower() == "y"
    _tui_done(f"De-bleed {'enabled' if debleed else 'disabled'}")
    return MaxModeOptions(mdx_model_key=mdx_key, dereverb_model_key=dr_key, debleed=debleed)


def _tui_check_dependencies() -> bool:
    if sys.version_info < (3, 9):
        print(f"Python 3.9+ required.  Current: {sys.version}")
        return False
    missing = _check_missing_packages()
    if missing:
        print(f"[setup] Missing packages: {', '.join(missing)}")
        ensure_dependencies()
        still_missing = _check_missing_packages()
        if still_missing:
            print(
                f"[setup] Still missing after install attempt: {', '.join(still_missing)}")
            print(
                "  Try: pip install demucs audio-separator onnxruntime torch soundfile numpy pyqt6")
            return False
    print(f"\n{_DIV}\n  Hardware Detection\n{_DIV}")
    backend, desc = HardwareDetector.get_best_backend()
    print(f"  {_GRAY}Detected{_RST}   {desc}")
    if backend == HardwareBackend.TENSORRT:
        _tui_info("TensorRT detected — engine cache will be built on first run")
    elif backend in (HardwareBackend.CUDA, HardwareBackend.ROCM):
        if not HardwareDetector.detect_onnxruntime_gpu():
            _tui_warn("MDX acceleration unavailable")
            if backend == HardwareBackend.CUDA:
                _tui_info("pip install onnxruntime-gpu")
    elif backend == HardwareBackend.MPS:
        _tui_info("MPS active — MDX models will run on CPU")
    elif backend == HardwareBackend.OPENVINO:
        if not HardwareDetector.detect_openvino():
            _tui_warn("pip install openvino onnxruntime-openvino")
    else:
        _tui_info("Running on CPU")
    vram = _get_vram_free_mb()
    ram = _get_ram_free_mb()
    if vram > 0:
        _tui_info(f"VRAM free: {vram:.0f} MB")
    _tui_info(f"RAM free:  {ram:.0f} MB")
    print(_DIV)
    return True


def _tui_collect_batch_files() -> list:
    print(f"\n{_DIV}\n  {_GRAY}Batch mode — enter file paths one by one, type 'done' to finish{_RST}\n")
    paths = []
    while True:
        raw = input(
            f"  {_TC}File {len(paths)+1} (or 'done'):{_RST} ").strip().strip('"').strip("'")
        if raw.lower() == "done":
            break
        if not raw:
            continue
        ap = Path(raw).expanduser()
        if not ap.exists():
            print(f"  {_GRAY}File not found: {ap}{_RST}")
            continue
        if ap.suffix.lower() not in AUDIO_EXTS:
            print(
                f"  {_GRAY}Unsupported format: {ap.suffix}  (supported: {', '.join(sorted(AUDIO_EXTS))}){_RST}")
            continue
        paths.append(ap)
        _tui_done(str(ap.name))
    return paths


def _tui_run():
    backend, backend_desc = HardwareDetector.get_best_backend()
    onnx_providers = HardwareDetector.get_onnx_providers(backend)
    os.system("cls" if os.name == "nt" else "clear")
    print(f"{_TC}")
    print(" ░▒▓███████▓▒░▒▓████████▓▒░▒▓████████▓▒░▒▓██████████████▓▒░░▒▓█▓▒░       ░▒▓██████▓▒░░▒▓███████▓▒░")
    print("░▒▓█▓▒░         ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░")
    print("░▒▓█▓▒░         ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░")
    print(" ░▒▓██████▓▒░   ░▒▓█▓▒░   ░▒▓██████▓▒░ ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓████████▓▒░▒▓███████▓▒░")
    print("       ░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░")
    print("       ░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░")
    print("░▒▓███████▓▒░   ░▒▓█▓▒░   ░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░░▒▓█▓▒░▒▓████████▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓███████▓▒░")
    print(_RST)
    print(f"  {_GRAY}Hardware       {_RST}{backend_desc}")
    print(f"  {_GRAY}ONNX Providers {_RST}{', '.join(onnx_providers)}")
    vram = _get_vram_free_mb()
    ram = _get_ram_free_mb()
    if vram > 0:
        print(f"  {_GRAY}VRAM Free      {_RST}{vram:.0f} MB")
    print(f"  {_GRAY}RAM Free       {_RST}{ram:.0f} MB")
    print(f"  {_GRAY}Data dir       {_RST}{_app_data_dir()}")
    if _TOML_OK and CONFIG_TOML.exists():
        print(
            f"  {_TC}[config.toml]{_RST} {_GRAY}Found — overrides active{_RST}")
    elif _TOML_OK:
        print(f"  {_GRAY}[config.toml]  Not found — using defaults{_RST}")
    else:
        print(f"  {_GRAY}[config.toml]  tomllib/tomli not available{_RST}")
    print(_DIV)
    print()
    _tui_opt("1", "Fast             — htdemucs · 1 shift · quick results")
    _tui_opt("2", "Fine-tuned       — htdemucs_ft · resource-aware shifts")
    _tui_opt("3", "Professional     — MDX + Demucs ensemble")
    _tui_opt("4", "Max              — full ensemble + post-processing")
    print()
    while True:
        choice = input(f"  {_TC}>{_RST} ").strip()
        if choice.lower() == "q":
            print(f"\n  {_GRAY}Goodbye!{_RST}\n")
            sys.exit(0)
        if choice in ("1", "2", "3", "4"):
            break
        print(f"  {_GRAY}Invalid — enter 1–4 or q to quit.{_RST}")

    preset = {"1": "fast", "2": "fine-tuned",
              "3": "professional", "4": "max"}[choice]
    max_options = _tui_prompt_max_options() if preset == "max" else None

    print(f"\n{_DIV}")
    seven_stem_raw = input(
        f"  7-Stem separation (guitar + piano stems)? {_GRAY}[y/N]{_RST}: ").strip().lower()
    seven_stem = seven_stem_raw == "y"
    if seven_stem:
        _tui_done("7-Stem mode enabled  (htdemucs_6s)")
        max_options = None

    print(f"\n{_DIV}\n  {_GRAY}Export format{_RST}\n")
    _tui_opt("1", "WAV  (lossless · 24-bit)  [default]")
    _tui_opt("2", "MP3  (320 kbps)")
    _tui_opt("3", "FLAC (lossless compressed)")
    fmt_raw = input(f"\n  {_TC}>{_RST} ").strip() or "1"
    export_fmt = "mp3" if fmt_raw == "2" else "flac" if fmt_raw == "3" else "wav"
    _tui_done(f"Export format: {export_fmt.upper()}")

    print(f"\n{_DIV}")
    batch_raw = input(
        f"  Batch processing? {_GRAY}[y/N]{_RST}: ").strip().lower()
    is_batch = batch_raw == "y"

    if is_batch:
        audio_paths = _tui_collect_batch_files()
        if not audio_paths:
            print(f"  {_GRAY}No valid files provided.  Exiting.{_RST}")
            sys.exit(0)
        batch_dir = Path.cwd() / "Stem_batch"
        batch_dir.mkdir(parents=True, exist_ok=True)
        _tui_done(f"Batch output: {batch_dir}")
    else:
        print(f"\n{_DIV}")
        while True:
            raw = input(f"  {_GRAY}Audio file path:{_RST} ").strip().strip(
                '"').strip("'")
            audio_path = Path(raw).expanduser()
            if audio_path.exists():
                if audio_path.suffix.lower() not in AUDIO_EXTS:
                    print(f"  {_GRAY}Unsupported format: {audio_path.suffix}"
                          f"  (supported: {', '.join(sorted(AUDIO_EXTS))}){_RST}")
                    continue
                break
            print(f"  {_GRAY}File not found: {audio_path}{_RST}")

        default_out = Path.cwd() / audio_path.stem
        raw_out = input(
            f"  {_GRAY}Output directory [{default_out}]:{_RST} ").strip()
        output_dir = Path(raw_out).expanduser() if raw_out else default_out

    cfg = SeparationConfig.from_preset(
        preset, backend, max_options=max_options,
        export_format=export_fmt,
        toml_overrides=_load_toml_overrides(preset),
        seven_stem=seven_stem)

    if is_batch:
        total = len(audio_paths)
        all_stems = {}
        for idx, ap in enumerate(audio_paths, 1):
            print(f"\n{_DIV}")
            print(f"  [{idx}/{total}]  {ap.name}")
            print(_DIV)
            out_dir = batch_dir / ap.stem
            separator = STEMSeparatorLogic(cfg)
            try:
                stems = separator.separate(ap, out_dir)
                _songs_json_append(ap.stem, out_dir, stems,
                                   preset=preset, model=cfg.model, backend_desc=backend_desc)
                _tui_done(f"{ap.name} — done")
                for name, path in stems.items():
                    p = Path(path)
                    mb = p.stat().st_size / (1024 * 1024) if p.exists() else 0
                    print(
                        f"    {_GRAY}{name.capitalize():14s}{_RST}  {p.name:26s}  {_GRAY}{mb:5.1f} MB{_RST}")
            except Exception as exc:
                print(f"  {_WHT}Error on {ap.name}:{_RST} {exc}")
                traceback.print_exc()
        print(f"\n{_DIV}")
        print(f"  {_TC}[✓]{_RST}  Batch complete  →  {batch_dir}")
        print(_DIV)
    else:
        separator = STEMSeparatorLogic(cfg)
        try:
            final_stems = separator.separate(audio_path, output_dir)
        except Exception as exc:
            print(f"\n  {_WHT}Error:{_RST} {exc}")
            traceback.print_exc()
            sys.exit(1)
        print(f"\n{_DIV}")
        print(f"  {_TC}[✓]{_RST}  Separation complete")
        print(_DIV)
        print()
        for name, path in final_stems.items():
            p = Path(path)
            mb = p.stat().st_size / (1024 * 1024) if p.exists() else 0
            print(
                f"  {_GRAY}{name.capitalize():14s}{_RST}  {p.name:26s}  {_GRAY}{mb:5.1f} MB{_RST}")
        print()


def main():
    if "--tui" in sys.argv:
        if not _tui_check_dependencies():
            sys.exit(1)
        _tui_run()
        return

    ensure_dependencies()
    run_gui()


if __name__ == "__main__":
    main()
