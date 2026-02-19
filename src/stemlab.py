"""
StemLab V1.0 - a free and powerful stem separation  
maintained at: https://github.com/purectl/StemLab
license: MIT
"""

from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtGui import (
    QFont, QColor, QPalette, QPainter, QPen, QBrush, QCursor,
)
from PyQt6.QtCore import (
    Qt, QThread, pyqtSignal, QPropertyAnimation, QEasingCurve,
    QTimer, pyqtProperty, QUrl,
)
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QComboBox, QCheckBox,
    QStackedWidget, QFrame, QProgressBar, QTextEdit, QScrollArea,
    QLineEdit, QSlider, QMessageBox, QRadioButton,
)
import os
import sys
import gc
import re
import shutil
import subprocess
import traceback
from pathlib import Path
from typing import Optional, Literal
from dataclasses import dataclass, field
from enum import Enum

os.environ.setdefault("QT_STYLE_OVERRIDE", "")
os.environ["QT_LOGGING_RULES"] = "qt.qpa.services=false"


BG = "#0a0a0a"
BG2 = "#111111"
BG3 = "#161616"
BORDER = "#1e1e1e"
BORDER2 = "#252525"
CYAN = "#00e5ff"
CYAN_DIM = "#00b8cc"
CYAN_GLOW = "#00e5ff33"
CYAN_DARK = "#003d4d"
TEXT = "#e8f4f8"
TEXT_DIM = "#6a8a90"
TEXT_MUTED = "#3a5a60"
ERROR_C = "#ff4444"

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


MDX_MODELS = {
    "MDX23C":            {"filename": "MDX23C-8KFFT-InstVoc_HQ.ckpt", "display": "MDX23C — Highest accuracy"},
    "Kim_Vocal_2":       {"filename": "Kim_Vocal_2.onnx",             "display": "Kim Vocal 2 — Clean isolation"},
    "UVR-MDX-NET-Voc_FT": {"filename": "UVR-MDX-NET-Voc_FT.onnx",    "display": "UVR-MDX-NET Vocal FT — Balanced"},
}

DEREVERB_MODELS = {
    "none":                {"filename": None,                      "display": "Disabled"},
    "UVR-DeEcho-DeReverb": {"filename": "UVR-DeEcho-DeReverb.pth", "display": "De-Echo / De-Reverb"},
    "UVR-DeNoise":         {"filename": "UVR-DeNoise.pth",         "display": "De-Noise (recommended)"},
}


def _get_vram_free_mb() -> float:
    if not TORCH_OK:
        return 0.0
    try:
        if torch.cuda.is_available():
            free, total = torch.cuda.mem_get_info(torch.cuda.current_device())
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
            if hasattr(torch.version, 'hip') and torch.version.hip:
                return True
            if torch.cuda.is_available():
                n = torch.cuda.get_device_name(0).lower()
                return 'amd' in n or 'radeon' in n
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
            return 'OpenVINOExecutionProvider' in ort.get_available_providers()
        except Exception:
            return False

    @staticmethod
    def detect_tensorrt() -> bool:
        try:
            import onnxruntime as ort
            return 'TensorrtExecutionProvider' in ort.get_available_providers()
        except Exception:
            return False

    @staticmethod
    def detect_onnxruntime_gpu() -> bool:
        try:
            import onnxruntime as ort
            gpu = {'CUDAExecutionProvider', 'ROCMExecutionProvider',
                   'OpenVINOExecutionProvider', 'TensorrtExecutionProvider'}
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
                import platform
                return HardwareBackend.MPS, f"Apple MPS  ({platform.processor() or 'Apple Silicon'})"
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
            return ['CPUExecutionProvider']
        if backend == HardwareBackend.TENSORRT and 'TensorrtExecutionProvider' in av:
            return ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
        if backend in (HardwareBackend.CUDA, HardwareBackend.TENSORRT) and 'CUDAExecutionProvider' in av:
            return ['CUDAExecutionProvider', 'CPUExecutionProvider']
        if backend == HardwareBackend.ROCM and 'ROCMExecutionProvider' in av:
            return ['ROCMExecutionProvider', 'CPUExecutionProvider']
        if backend == HardwareBackend.OPENVINO and 'OpenVINOExecutionProvider' in av:
            return ['OpenVINOExecutionProvider', 'CPUExecutionProvider']
        return ['CPUExecutionProvider']


@dataclass
class MaxModeOptions:
    mdx_model_key:      str = "MDX23C"
    dereverb_model_key: Optional[str] = "UVR-DeNoise"
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

    @classmethod
    def from_preset(cls, preset: str, backend=None, max_options=None, export_format="wav"):
        if backend is None:
            backend, _ = HardwareDetector.get_best_backend()
        device = ("cuda" if backend in (HardwareBackend.CUDA, HardwareBackend.ROCM, HardwareBackend.TENSORRT)
                  else "mps" if backend == HardwareBackend.MPS else "cpu")
        onnx = HardwareDetector.get_onnx_providers(backend)
        gpu = backend in (HardwareBackend.CUDA,
                          HardwareBackend.ROCM, HardwareBackend.TENSORRT)

        raw_shifts_gpu, raw_shifts_cpu = _resource_aware_shifts(preset, gpu)

        table = {
            "fast":         dict(model="htdemucs",    shifts=1,              overlap=0.25, use_mdx_vocals=False),
            "fine-tuned":   dict(model="htdemucs_ft", shifts=raw_shifts_cpu, overlap=0.25, use_mdx_vocals=False),
            "professional": dict(model="htdemucs_ft", shifts=raw_shifts_gpu if gpu else raw_shifts_cpu,
                                 overlap=0.5, use_mdx_vocals=True),
            "max":          dict(model="htdemucs_ft", shifts=raw_shifts_gpu if gpu else raw_shifts_cpu,
                                 overlap=0.5, use_mdx_vocals=True),
        }
        kw = table.get(preset, table["fine-tuned"])
        cfg = cls(quality_preset=preset, device=device, backend=backend, onnx_providers=onnx,
                  max_options=max_options or MaxModeOptions(), export_format=export_format, **kw)
        if backend == HardwareBackend.CPU and not HardwareDetector.detect_onnxruntime_gpu():
            cfg.use_mdx_vocals = False
        return cfg


def _resource_aware_shifts(preset: str, gpu: bool) -> tuple:
    vram_free = _get_vram_free_mb()
    if vram_free > 0:
        if vram_free >= 8192:
            gpu_shifts = 4
        elif vram_free >= 4096:
            gpu_shifts = 3
        elif vram_free >= 2048:
            gpu_shifts = 2
        else:
            gpu_shifts = 1
    else:
        gpu_shifts = 4  

    ram_free = _get_ram_free_mb()
    if ram_free >= 8192:
        cpu_shifts = 2
    elif ram_free >= 4096:
        cpu_shifts = 1
    else:
        cpu_shifts = 1

    if preset in ("fast", "fine-tuned"):
        gpu_shifts = min(gpu_shifts, 2)

    return gpu_shifts, cpu_shifts


def _phase_aware_merge(stems: list, samplerate: int) -> "np.ndarray":
    try:
        import numpy as _np
        import soundfile as _sf  
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
    window = _np.sqrt(_np.hanning(win_size)).astype(_np.float64)
    n_samp = min_len

    out = _np.zeros((n_samp, n_ch), dtype=_np.float64)
    n_acc = _np.zeros(n_samp,         dtype=_np.float64)

    combined_specs = None

    for arr in arrays:
        if arr.shape[1] < n_ch:
            arr = _np.repeat(arr, n_ch, axis=1)

        specs = []
        for ch in range(n_ch):
            sig = arr[:, ch].astype(_np.float64)
            spec = _np.array([
                _np.fft.rfft(sig[pos:pos + win_size] * window)
                for pos in range(0, n_samp - win_size + 1, hop)
            ])
            specs.append(spec)
        specs = _np.stack(specs, axis=0) 

        if combined_specs is None:
            combined_specs = specs
        else:
            combined_specs = combined_specs + specs

    if combined_specs is None:
        return _np.zeros((min_len, n_ch), dtype=_np.float32)

    for ch in range(n_ch):
        out_ch = _np.zeros(n_samp, dtype=_np.float64)
        acc_ch = _np.zeros(n_samp, dtype=_np.float64)
        for fi, pos in enumerate(range(0, n_samp - win_size + 1, hop)):
            frame = _np.fft.irfft(combined_specs[ch, fi], n=win_size)
            out_ch[pos:pos + win_size] += frame * window
            acc_ch[pos:pos + win_size] += window ** 2
        out[:, ch] = out_ch / _np.where(acc_ch > 1e-8, acc_ch, 1.0)

    peak = _np.max(_np.abs(out))
    if peak > 0.98:
        out = out / peak * 0.98

    return out.astype(_np.float32)


class STEMSeparatorLogic:
    _VR_EXT = {".pth"}
    _MDX_EXT = {".onnx"}
    _MDXC_EXT = {".ckpt"}

    def __init__(self, config: SeparationConfig, progress_cb=None):
        self.config = config
        self.progress_cb = progress_cb or (lambda _: None)
        self.demucs_model = None

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
            print(
                f"  [mem] VRAM free after flush: {_get_vram_free_mb():.0f} MB")
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
                model_filename,
                sess_options=trt_opts,
                providers=[("TensorrtExecutionProvider", provider_opts),
                           "CUDAExecutionProvider",
                           "CPUExecutionProvider"],
            )
            print(f"  [trt] Engine cache written to {cache_dir}")
        except Exception as e:
            print(f"  [trt] Optimisation skipped: {e}")
        return model_filename

    def _load_demucs(self):
        if self.demucs_model:
            return
        from demucs.pretrained import get_model
        print(f"▸ Loading Demucs [{self.config.model}]")
        self.demucs_model = get_model(self.config.model)
        self.demucs_model.to(self.config.device)
        self.demucs_model.eval()
        if self.config.backend in (HardwareBackend.CUDA, HardwareBackend.TENSORRT):
            torch.backends.cudnn.benchmark = True
        elif self.config.backend == HardwareBackend.OPENVINO:
            torch.set_num_threads(os.cpu_count() or 4)
        print(f"✓ Demucs ready on {self.config.device.upper()}")

    def _stem_ext(self):
        return "mp3" if self.config.export_format == "mp3" else "wav"

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

    def _save_audio(self, src, name: str, out_dir: Path, sr: int) -> Path:
        from demucs.audio import save_audio
        wav_p = out_dir / f"{name}.wav"
        save_audio(src, str(wav_p), samplerate=sr, bitrate=320,
                   clip='rescale', as_float=False, bits_per_sample=24)
        if self._stem_ext() == "mp3":
            mp3_p = out_dir / f"{name}.mp3"
            ok = self._wav_to_mp3(wav_p, mp3_p)
            wav_p.unlink(missing_ok=True)
            if ok:
                print(f"✓ Saved {mp3_p.name}")
                return mp3_p
            keep = out_dir / f"{name}.wav"
            print(f"⚠ MP3 encode failed — saved as WAV (install ffmpeg or pydub)")
            return keep
        print(f"✓ Saved {wav_p.name}")
        return wav_p

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
        return {name: self._save_audio(sources[i], name, output_dir, sr)
                for i, name in enumerate(["drums", "bass", "other", "vocals"])}

    def _set_ort_env(self):
        m = {
            HardwareBackend.CUDA:      'CUDAExecutionProvider',
            HardwareBackend.TENSORRT:  'TensorrtExecutionProvider',
            HardwareBackend.ROCM:      'ROCMExecutionProvider',
            HardwareBackend.OPENVINO:  'OpenVINOExecutionProvider',
        }
        ep = m.get(self.config.backend)
        if ep:
            os.environ['ORT_EXECUTION_PROVIDER'] = ep

    def _make_separator(self, model_filename: str):
        from audio_separator.separator import Separator
        self._set_ort_env()
        self._maybe_free_gpu()

        model_filename = self._apply_tensorrt_optimisation(model_filename)

        print(f"  Loading {model_filename}")
        ext = Path(model_filename).suffix.lower()
        kw = dict(model_file_dir="/tmp/audio_separator_models", output_format="wav",
                  normalization_threshold=0.9, amplification_threshold=0.9)
        if ext in self._VR_EXT:
            kw["vr_params"] = dict(batch_size=8, window_size=512, aggression=5, enable_tta=False,
                                   enable_post_process=False, post_process_threshold=0.2, high_end_process=False)
        elif ext in self._MDXC_EXT:
            kw["mdxc_params"] = dict(
                segment_size=256, overlap=8, batch_size=1, pitch_shift=0)
        else:
            kw["mdx_params"] = dict(
                hop_length=1024, segment_size=256, overlap=0.25, batch_size=1)
        sep = Separator(**kw)
        sep.load_model(model_filename=model_filename)
        return sep

    def separate_vocals_mdx(self, audio_path: Path, output_dir: Path, model_filename: str) -> Path:
        print(f"▸ MDX vocal extraction [{model_filename}]")
        sep = self._make_separator(model_filename)
        files = sep.separate(str(audio_path))
        vf = next((Path(f)
                  for f in files if "Vocals" in f or "vocals" in f), None)
        if vf is None:
            raise RuntimeError("MDX-NET did not produce a vocal stem")
        target = output_dir / f"vocals.{self._stem_ext()}"
        if vf.resolve() != target.resolve():
            shutil.move(str(vf), str(target))
        print(f"✓ Saved {target.name}")
        for f in files:
            p = Path(f)
            if p.exists() and p.resolve() != target.resolve():
                p.unlink(missing_ok=True)
        return target

    def apply_dereverb(self, vocal_path: Path, output_dir: Path, model_filename: str) -> Path:
        print(f"▸ Post-processing [{model_filename}]")
        sep = self._make_separator(model_filename)
        files = sep.separate(str(vocal_path))
        if not files:
            print("⚠ No output — skipping.")
            return vocal_path
        CLEAN = ["(no noise)", "(no echo)", "(no reverb)",
                 "(vocals)", "no noise", "no echo", "no reverb"]
        existing = [Path(f) for f in files if Path(f).exists()]
        cleaned = next(
            (Path(f) for term in CLEAN for f in files if term in Path(f).name.lower()), None)
        if cleaned is None or not cleaned.exists():
            cleaned = min(existing, key=lambda p: p.stat().st_size)
        target = output_dir / f"vocals.{self._stem_ext()}"
        if cleaned.resolve() != target.resolve():
            shutil.move(str(cleaned), str(target))
        print(f"✓ Saved {target.name} (post-processed)")
        for p in existing:
            if p.exists() and p.resolve() != target.resolve():
                p.unlink(missing_ok=True)
        return target

    def apply_debleed(self, vocal_path: Path, instrumental_path: Path, output_dir: Path) -> Path:
        import soundfile as sf
        print("▸ De-bleed")
        vocals, sr_v = sf.read(str(vocal_path))
        instrum, sr_i = sf.read(str(instrumental_path))
        if sr_v != sr_i:
            print("⚠ SR mismatch — skipping.")
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
        ws = 2048
        hop = ws // 4
        alpha = 0.03
        win = np.sqrt(np.hanning(ws)).astype(np.float64)
        ns = vocals.shape[0]
        result = np.zeros_like(vocals, dtype=np.float64)
        for ch in range(vocals.shape[1]):
            v = vocals[:, ch].astype(np.float64)
            i = instrum[:, ch].astype(np.float64)
            out = np.zeros(ns, dtype=np.float64)
            nacc = np.zeros(ns, dtype=np.float64)
            for pos in range(0, ns - ws + 1, hop):
                V = np.fft.rfft(v[pos:pos + ws] * win)
                I = np.fft.rfft(i[pos:pos + ws] * win)
                Vp = np.abs(V) ** 2
                Ip = np.abs(I) ** 2
                mask = Vp / np.maximum(Vp + alpha * Ip, 1e-12)
                out[pos:pos + ws] += np.fft.irfft(V * mask, n=ws) * win
                nacc[pos:pos + ws] += win ** 2
            result[:, ch] = out / np.where(nacc > 1e-8, nacc, 1.0)
        peak = np.max(np.abs(result))
        if peak <= 1e-6:
            print("⚠ Silent — keeping original.")
            return vocal_path
        if peak > 0.98:
            result = result / peak * 0.98
        if self._stem_ext() == "mp3":
            tmp = output_dir / "vocals_debleed_tmp.wav"
            import soundfile as sf2
            sf2.write(str(tmp), result.astype(
                np.float32), sr_v, subtype='PCM_24')
            target = output_dir / "vocals.mp3"
            ok = self._wav_to_mp3(tmp, target)
            tmp.unlink(missing_ok=True)
            if not ok:
                target = output_dir / "vocals.wav"
                sf2.write(str(target), result.astype(
                    np.float32), sr_v, subtype='PCM_24')
                print("⚠ MP3 encode failed — saved as WAV")
        else:
            import soundfile as sf2
            target = output_dir / "vocals.wav"
            sf2.write(str(target), result.astype(
                np.float32), sr_v, subtype='PCM_24')
        print(f"✓ Saved {target.name} (de-bled)")
        return target

    def mix_stems(self, paths: list, out: Path, sr: int):
        import soundfile as sf
        print(f"▸ merge → {out.name}")
        arrays = []
        for p in paths:
            a, _ = sf.read(str(p))
            arrays.append(a)

        merged = _phase_aware_merge(arrays, sr)
        if merged is None:
            print("⚠ Merge produced no data — output will be silent.")
            import numpy as _np
            merged = _np.zeros((sr * 2, 2), dtype=_np.float32)

        if out.suffix.lower() == ".mp3":
            tmp = out.with_suffix(".mix_tmp.wav")
            sf.write(str(tmp), merged, sr, subtype='PCM_24')
            ok = self._wav_to_mp3(tmp, out)
            tmp.unlink(missing_ok=True)
            if not ok:
                out = out.with_suffix(".wav")
                sf.write(str(out), merged, sr, subtype='PCM_24')
                print("⚠ MP3 encode failed — saved as WAV")
        else:
            sf.write(str(out), merged, sr, subtype='PCM_24')
        print(f"✓ Saved {out.name}")

    def separate(self, audio_path: Path, output_dir: Path) -> dict:
        audio_path = Path(audio_path)
        output_dir = Path(output_dir)
        if not audio_path.exists():
            raise FileNotFoundError(f"Not found: {audio_path}")
        output_dir.mkdir(parents=True, exist_ok=True)
        self.progress_cb(5)
        ext = self._stem_ext()

        if not self.config.use_mdx_vocals:
            print(f"▸ {self.config.quality_preset.upper()} — Demucs")
            stems = self.separate_with_demucs(audio_path, output_dir)
            self.progress_cb(70)
            inst = output_dir / f"instrumental.{ext}"
            self.mix_stems([stems["bass"], stems["drums"], stems["other"]], inst,
                           self.demucs_model.samplerate)
            self.progress_cb(100)
            return {**stems, "instrumental": inst}
        else:
            is_max = self.config.quality_preset == "max"
            opts = self.config.max_options if is_max else None
            mdx_fn = MDX_MODELS[opts.mdx_model_key if is_max else "UVR-MDX-NET-Voc_FT"]["filename"]
            print(f"▸ {'MAX' if is_max else 'PROFESSIONAL'} — Ensemble")
            mdx_vox = self.separate_vocals_mdx(audio_path, output_dir, mdx_fn)
            self.progress_cb(30)
            stems = self.separate_with_demucs(audio_path, output_dir)
            self.progress_cb(65)
            self._free_gpu(force=True)
            inst = output_dir / f"instrumental.{ext}"
            self.mix_stems([stems["bass"], stems["drums"], stems["other"]], inst,
                           self.demucs_model.samplerate)
            self.progress_cb(75)
            if is_max and opts.dereverb_model_key:
                mdx_vox = self.apply_dereverb(
                    mdx_vox, output_dir, DEREVERB_MODELS[opts.dereverb_model_key]["filename"])
            self.progress_cb(88)
            if is_max and opts.debleed:
                mdx_vox = self.apply_debleed(mdx_vox, inst, output_dir)
            self.progress_cb(100)
            return {"vocals": mdx_vox, "bass": stems["bass"], "drums": stems["drums"],
                    "other": stems["other"], "instrumental": inst}


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
                        clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
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



def _sec_lbl(text: str) -> QLabel:
    l = QLabel(text)
    l.setStyleSheet(
        f"color:{TEXT_MUTED};font-size:10px;letter-spacing:3px;background:transparent;")
    return l


class GlowButton(QPushButton):
    def __init__(self, text, parent=None, primary=False, danger=False):
        super().__init__(text, parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        c = ERROR_C if danger else CYAN
        dim = "#cc3333" if danger else CYAN_DIM
        glow = "#ff444433" if danger else CYAN_GLOW
        dark = "#3d0000" if danger else CYAN_DARK
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
        cx = self.width() // 2
        cy = self.height() // 2
        r = self._r
        p.drawEllipse(int(cx - r), int(cy - r), int(r * 2), int(r * 2))


STEM_COLOURS = {"vocals": "#00e5ff", "bass": "#00c8e0", "drums": "#00aabf",
                "other": "#008fa0", "instrumental": "#007280"}
STEM_ICONS = {"vocals": "◈", "bass": "◉",
              "drums": "◎", "other": "◐", "instrumental": "◑"}


class StemPlayer(QFrame):
    def __init__(self, stem_name: str, file_path: str, parent=None):
        super().__init__(parent)
        self.stem_name = stem_name
        self.file_path = file_path
        self._seeking = False
        self._duration = 0
        colour = STEM_COLOURS.get(stem_name, CYAN)
        self.setStyleSheet(
            f"StemPlayer{{background:{BG2};border:1px solid {BORDER};border-radius:9px;}}")

        self._player = QMediaPlayer()
        self._audio = QAudioOutput()
        self._player.setAudioOutput(self._audio)
        self._audio.setVolume(0.9)
        self._player.setSource(QUrl.fromLocalFile(str(file_path)))
        self._player.positionChanged.connect(self._on_pos)
        self._player.durationChanged.connect(self._on_dur)
        self._player.playbackStateChanged.connect(self._on_state)

        root = QVBoxLayout(self)
        root.setContentsMargins(18, 14, 18, 14)
        root.setSpacing(10)

        top = QHBoxLayout()
        top.setSpacing(12)
        icon_l = QLabel(STEM_ICONS.get(stem_name, "▸"))
        icon_l.setStyleSheet(f"color:{colour};font-size:20px;")
        icon_l.setFixedWidth(22)
        name_l = QLabel(stem_name.upper())
        name_l.setStyleSheet(
            f"color:{colour};font-size:12px;font-weight:bold;letter-spacing:2px;")
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

        ctrl = QHBoxLayout()
        ctrl.setSpacing(10)
        self._play_btn = QPushButton("▶")
        self._play_btn.setFixedSize(34, 34)
        self._play_btn.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._play_btn.setStyleSheet(f"""
            QPushButton {{
                background:{CYAN_DARK}; border:1px solid {colour};
                border-radius:17px; color:{colour}; font-size:13px;
            }}
            QPushButton:hover   {{ background:{colour}; color:#000; }}
            QPushButton:pressed {{ background:{CYAN_DIM}; }}
        """)
        self._play_btn.clicked.connect(self._toggle)

        self._pos_l = QLabel("0:00")
        self._pos_l.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
        self._pos_l.setFixedWidth(36)

        self._seek = QSlider(Qt.Orientation.Horizontal)
        self._seek.setRange(0, 1000)
        self._seek.sliderPressed.connect(
            lambda: setattr(self, '_seeking', True))
        self._seek.sliderReleased.connect(self._on_seek_release)
        self._seek.setStyleSheet(f"""
            QSlider::groove:horizontal {{ background:{BORDER2}; height:3px; border-radius:1px; }}
            QSlider::handle:horizontal {{
                background:{colour}; width:12px; height:12px; border-radius:6px; margin:-5px 0;
            }}
            QSlider::sub-page:horizontal {{ background:{colour}; border-radius:1px; }}
        """)

        self._dur_l = QLabel("0:00")
        self._dur_l.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
        self._dur_l.setFixedWidth(36)
        self._dur_l.setAlignment(
            Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)

        ctrl.addWidget(self._play_btn)
        ctrl.addWidget(self._pos_l)
        ctrl.addWidget(self._seek, stretch=1)
        ctrl.addWidget(self._dur_l)
        root.addLayout(ctrl)

    @staticmethod
    def _fmt(ms: int) -> str:
        s = ms // 1000
        return f"{s // 60}:{s % 60:02d}"

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
                self._seek.setValue(int(pos / self._duration * 1000))

    def _on_dur(self, dur: int):
        self._duration = dur
        self._dur_l.setText(self._fmt(dur))

    def _on_seek_release(self):
        self._seeking = False
        if self._duration > 0:
            self._player.setPosition(
                int(self._seek.value() / 1000 * self._duration))

    def stop_playback(self): self._player.stop()
    def cleanup(self):       self._player.stop(
    ); self._player.setSource(QUrl())


class StemFolderCard(QFrame):
    clicked = pyqtSignal(str, str)

    def __init__(self, stem_name: str, file_path: str, parent=None):
        super().__init__(parent)
        self.stem_name = stem_name
        self.file_path = file_path
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._colour = STEM_COLOURS.get(stem_name, CYAN)
        self.setFixedSize(148, 130)
        self._refresh(False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 16, 12, 12)
        lay.setSpacing(7)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)

        icon = QLabel(STEM_ICONS.get(stem_name, "▸"))
        icon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        icon.setStyleSheet(f"color:{self._colour};font-size:34px;")

        name = QLabel(stem_name.upper())
        name.setAlignment(Qt.AlignmentFlag.AlignCenter)
        name.setStyleSheet(
            f"color:{self._colour};font-size:11px;font-weight:bold;letter-spacing:2px;")

        p = Path(file_path)
        sz = QLabel(
            f"{p.stat().st_size/1_048_576:.1f} MB" if p.exists() else "")
        sz.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sz.setStyleSheet(f"color:{TEXT_MUTED};font-size:10px;")

        lay.addStretch()
        lay.addWidget(icon)
        lay.addWidget(name)
        lay.addWidget(sz)
        lay.addStretch()

    def _refresh(self, hover: bool):
        c = self._colour if hover else BORDER2
        bg = "#0d1a1e" if hover else BG2
        self.setStyleSheet(
            f"StemFolderCard{{background:{bg};border:1px solid {c};border-radius:10px;}}")

    def enterEvent(self, e): self._refresh(True);  super().enterEvent(e)
    def leaveEvent(self, e): self._refresh(False); super().leaveEvent(e)
    def mousePressEvent(self, e): self.clicked.emit(
        self.stem_name, self.file_path); super().mousePressEvent(e)


class SongCard(QFrame):
    clicked = pyqtSignal()

    def __init__(self, song_name: str, out_dir: str, n_stems: int, parent=None):
        super().__init__(parent)
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self.setFixedHeight(130)
        self._idle_style = f"SongCard{{background:{BG2};border:1px solid {BORDER};border-radius:10px;}}"
        self._hover_style = f"SongCard{{background:#0d1a1e;border:1px solid {CYAN};border-radius:10px;}}"
        self.setStyleSheet(self._idle_style)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(24, 18, 24, 18)
        lay.setSpacing(8)

        row = QHBoxLayout()
        row.setSpacing(12)
        icon = QLabel("◫")
        icon.setStyleSheet(f"color:{CYAN};font-size:28px;")
        icon.setFixedWidth(36)
        title = QLabel(song_name.upper())
        title.setStyleSheet(
            f"color:{CYAN};font-size:18px;font-weight:bold;letter-spacing:3px;")
        badge = QLabel(f"{n_stems} stems")
        badge.setStyleSheet(f"color:{TEXT_MUTED};font-size:12px;")
        arrow = QLabel("›")
        arrow.setStyleSheet(f"color:{TEXT_MUTED};font-size:22px;")
        arrow.setFixedWidth(18)
        row.addWidget(icon)
        row.addWidget(title, stretch=1)
        row.addWidget(badge)
        row.addSpacing(8)
        row.addWidget(arrow)

        path_lbl = QLabel(out_dir)
        path_lbl.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
        lay.addLayout(row)
        lay.addWidget(path_lbl)

    def enterEvent(self, e):  self.setStyleSheet(
        self._hover_style); super().enterEvent(e)
    def leaveEvent(self, e):  self.setStyleSheet(
        self._idle_style);  super().leaveEvent(e)
    def mousePressEvent(self, e): self.clicked.emit(
    ); super().mousePressEvent(e)


class StemRowCard(QFrame):
    clicked = pyqtSignal(str, str)

    def __init__(self, stem_name: str, file_path: str, parent=None):
        super().__init__(parent)
        self.stem_name = stem_name
        self.file_path = file_path
        self.setCursor(QCursor(Qt.CursorShape.PointingHandCursor))
        self._colour = STEM_COLOURS.get(stem_name, CYAN)
        self.setFixedHeight(64)
        self._refresh(False)

        lay = QHBoxLayout(self)
        lay.setContentsMargins(20, 0, 20, 0)
        lay.setSpacing(16)

        icon = QLabel(STEM_ICONS.get(stem_name, "▸"))
        icon.setStyleSheet(f"color:{self._colour};font-size:20px;")
        icon.setFixedWidth(24)
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
        sz_lbl.setAlignment(Qt.AlignmentFlag.AlignRight |
                            Qt.AlignmentFlag.AlignVCenter)
        arrow = QLabel("›")
        arrow.setStyleSheet(f"color:{TEXT_MUTED};font-size:18px;")
        arrow.setFixedWidth(14)

        lay.addWidget(icon)
        lay.addWidget(name_lbl)
        lay.addWidget(file_lbl, stretch=1)
        lay.addWidget(sz_lbl)
        lay.addSpacing(8)
        lay.addWidget(arrow)

    def _refresh(self, hover: bool):
        c = self._colour if hover else BORDER
        bg = "#0d1a1e" if hover else BG2
        self.setStyleSheet(
            f"StemRowCard{{background:{bg};border:1px solid {c};border-radius:8px;}}")

    def enterEvent(self, e): self._refresh(True);  super().enterEvent(e)
    def leaveEvent(self, e): self._refresh(False); super().leaveEvent(e)

    def mousePressEvent(self, e):
        self.clicked.emit(self.stem_name, self.file_path)
        super().mousePressEvent(e)



class StemLabWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("STEMLAB")
        self.setMinimumSize(860, 680)
        self.resize(1020, 780)
        self.backend, self.backend_desc = HardwareDetector.get_best_backend()
        self.onnx_providers = HardwareDetector.get_onnx_providers(self.backend)
        self.worker: Optional[SeparationWorker] = None
        self._stems: dict = {}
        self._out_dir: Optional[Path] = None
        self._song_name: str = ""
        self._active_players: list = []
        self._build_ui()
        self.setStyleSheet(STYLE)

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
        self._stack.addWidget(self._build_stems_list_page())   
        self._stack.addWidget(self._build_stem_player_page())  
        body.addWidget(self._stack, stretch=1)
        rl.addLayout(body, stretch=1)

    def _build_header(self) -> QWidget:
        h = QWidget()
        h.setFixedHeight(54)
        h.setStyleSheet(f"background:{BG};border-bottom:1px solid {BORDER};")
        lay = QHBoxLayout(h)
        lay.setContentsMargins(24, 0, 24, 0)
        logo = QLabel("STEM<span style='color:{c}'>LAB</span>".format(c=CYAN))
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
        s.setStyleSheet(f"background:{BG2};border-right:1px solid {BORDER};")
        lay = QVBoxLayout(s)
        lay.setContentsMargins(0, 24, 0, 24)
        lay.setSpacing(2)
        self._nav_btns = []
        for num, label, idx in [("01", "CONFIGURE", 0), ("02", "PROCESS", 1), ("03", "RESULTS", 2)]:
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
        ver = QLabel("v1.0  ·  StemLab")
        ver.setStyleSheet(f"color:{TEXT_MUTED};font-size:10px;padding:0 16px;")
        lay.addWidget(ver)
        return s

    def _nav_to(self, idx: int):
        for i, b in enumerate(self._nav_btns):
            b.setChecked(i == min(idx, 2))
        self._stack.setCurrentIndex(idx)

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

        self._max_panel = QFrame()
        self._max_panel.setStyleSheet(
            f"background:{BG2};border:1px solid {BORDER};border-radius:8px;")
        mp = QVBoxLayout(self._max_panel)
        mp.setContentsMargins(20, 16, 20, 16)
        mp.setSpacing(14)
        mp.addWidget(_sec_lbl("MAX MODE OPTIONS"))

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
        for k, v in MDX_MODELS.items():
            self._mdx_combo.addItem(v["display"], k)
        mp.addLayout(_opt_row("Vocal model", self._mdx_combo))

        self._dr_combo = QComboBox()
        for k, v in DEREVERB_MODELS.items():
            self._dr_combo.addItem(v["display"], k)
        self._dr_combo.setCurrentIndex(2)
        mp.addLayout(_opt_row("Post-process", self._dr_combo))

        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"color:{BORDER};")
        mp.addWidget(div)

        self._debleed_chk = QCheckBox(
            "De-bleed  —  attenuate rhythmic bleed (snare / hi-hat) from instrumental")
        self._debleed_chk.setStyleSheet(f"color:{TEXT_DIM};font-size:12px;")
        mp.addWidget(self._debleed_chk)
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
        self._fmt_wav.setChecked(True)
        self._fmt_mp3 = QRadioButton("MP3  (320 kbps)")
        for rb in (self._fmt_wav, self._fmt_mp3):
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

        def _io_row(label, edit, btn):
            r = QHBoxLayout()
            r.setSpacing(8)
            l = QLabel(label)
            l.setStyleSheet(f"color:{TEXT_MUTED};font-size:12px;")
            l.setFixedWidth(56)
            r.addWidget(l)
            r.addWidget(edit, stretch=1)
            r.addWidget(btn)
            return r

        self._audio_edit = QLineEdit()
        self._audio_edit.setPlaceholderText(
            "Drag & drop or browse for audio file…")
        self._audio_edit.textChanged.connect(self._update_start_btn)
        in_btn = GlowButton("Browse")
        in_btn.clicked.connect(self._browse_audio)
        il.addLayout(_io_row("Input", self._audio_edit, in_btn))

        self._out_edit = QLineEdit()
        self._out_edit.setPlaceholderText(
            "Output directory (auto-filled from input)")
        out_btn = GlowButton("Browse")
        out_btn.clicked.connect(self._browse_output)
        il.addLayout(_io_row("Output", self._out_edit, out_btn))
        lay.addWidget(io)
        lay.addStretch()

        br = QHBoxLayout()
        br.addStretch()
        self._start_btn = GlowButton("▶  START SEPARATION", primary=True)
        self._start_btn.setFixedHeight(44)
        self._start_btn.setEnabled(False)
        self._start_btn.clicked.connect(self._start)
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

    def _on_preset_click(self, key: str):
        self._selected_preset = key
        for k, c in self._preset_cards.items():
            c.set_active(k == key)
        self._max_panel.setVisible(key == "max")

    def _browse_audio(self):
        p, _ = QFileDialog.getOpenFileName(self, "Select Audio File", "",
                                           "Audio (*.wav *.mp3 *.flac *.ogg *.aiff *.m4a);;All (*)")
        if p:
            self._audio_edit.setText(p)
            if not self._out_edit.text():
                self._out_edit.setText(str(Path(p).parent / Path(p).stem))

    def _browse_output(self):
        p = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if p:
            self._out_edit.setText(p)

    def _update_start_btn(self):
        self._start_btn.setEnabled(bool(self._audio_edit.text().strip()))

    def dragEnterEvent(self, e):
        if e.mimeData().hasUrls():
            e.acceptProposedAction()

    def dropEvent(self, e):
        urls = e.mimeData().urls()
        if urls:
            p = urls[0].toLocalFile()
            self._audio_edit.setText(p)
            if not self._out_edit.text():
                self._out_edit.setText(str(Path(p).parent / Path(p).stem))

    def _build_running_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(32, 28, 32, 28)
        lay.setSpacing(14)
        sr = QHBoxLayout()
        self._pulse = PulsingDot()
        self._status_lbl = QLabel("Initialising…")
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
        hint = QLabel("Click the song to browse its stems")
        hint.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
        lay.addWidget(hint)
        self._song_card_area = QWidget()
        self._song_card_area.setStyleSheet("background:transparent;")
        self._song_card_lay = QVBoxLayout(self._song_card_area)
        self._song_card_lay.setContentsMargins(0, 8, 0, 8)
        self._song_card_lay.setSpacing(0)
        lay.addWidget(self._song_card_area)
        lay.addStretch()
        bot = QHBoxLayout()
        bot.addStretch()
        open_btn = GlowButton("Open Output Folder")
        open_btn.clicked.connect(self._open_folder)
        bot.addWidget(open_btn)
        lay.addLayout(bot)
        return w

    def _build_stems_list_page(self) -> QWidget:
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(32, 28, 32, 28)
        lay.setSpacing(16)
        top = QHBoxLayout()
        back_btn = GlowButton("← Results")
        back_btn.clicked.connect(lambda: self._nav_to(2))
        self._stems_list_title = QLabel()
        self._stems_list_title.setStyleSheet(
            f"color:{CYAN};font-size:15px;font-weight:bold;letter-spacing:3px;")
        self._stems_list_path = QLabel()
        self._stems_list_path.setStyleSheet(
            f"color:{TEXT_MUTED};font-size:11px;")
        top.addWidget(back_btn)
        top.addSpacing(16)
        top.addWidget(self._stems_list_title)
        top.addSpacing(10)
        top.addWidget(self._stems_list_path, stretch=1)
        lay.addLayout(top)
        div = QFrame()
        div.setFrameShape(QFrame.Shape.HLine)
        div.setStyleSheet(f"color:{BORDER};")
        lay.addWidget(div)
        hint = QLabel("Click a stem to listen")
        hint.setStyleSheet(f"color:{TEXT_MUTED};font-size:11px;")
        lay.addWidget(hint)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background:transparent;border:none;")
        self._stems_list_w = QWidget()
        self._stems_list_w.setStyleSheet("background:transparent;")
        self._stems_list_lay = QVBoxLayout(self._stems_list_w)
        self._stems_list_lay.setContentsMargins(0, 0, 0, 0)
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
        back_btn = GlowButton("← Stems")
        back_btn.clicked.connect(
            lambda: (self._stop_players(), self._nav_to(3)))
        self._player_page_title = QLabel()
        self._player_page_title.setStyleSheet(
            f"color:{CYAN};font-size:15px;font-weight:bold;letter-spacing:2px;")
        self._player_page_song = QLabel()
        self._player_page_song.setStyleSheet(
            f"color:{TEXT_MUTED};font-size:12px;")
        top.addWidget(back_btn)
        top.addSpacing(16)
        top.addWidget(self._player_page_title)
        top.addSpacing(10)
        top.addWidget(self._player_page_song, stretch=1)
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

    def _start(self):
        ap = Path(self._audio_edit.text().strip())
        ot = self._out_edit.text().strip()
        od = Path(ot) if ot else (ap.parent / ap.stem)
        if not ap.exists():
            QMessageBox.warning(self, "Not Found", f"File not found:\n{ap}")
            return

        preset = self._selected_preset
        fmt = "mp3" if self._fmt_mp3.isChecked() else "wav"
        max_opts = None
        if preset == "max":
            dr_key = self._dr_combo.currentData()
            max_opts = MaxModeOptions(
                mdx_model_key=self._mdx_combo.currentData(),
                dereverb_model_key=None if dr_key == "none" else dr_key,
                debleed=self._debleed_chk.isChecked(),
            )

        cfg = SeparationConfig.from_preset(
            preset, self.backend, max_opts, export_format=fmt)
        self._log.clear()
        self._progress.setValue(0)
        self._status_lbl.setText("Starting…")
        self._pulse.start()
        self._nav_to(1)
        self._out_dir = od
        self._stems = {}
        self._song_name = ap.stem

        self.worker = SeparationWorker(cfg, ap, od)
        self.worker.log_line.connect(self._on_log)
        self.worker.progress.connect(self._progress.setValue)
        self.worker.stem_ready.connect(lambda n, p: self._stems.update({n: p}))
        self.worker.finished.connect(self._on_finished)
        self.worker.start()

    def _on_log(self, text: str):
        self._log.append(f'<span style="color:{TEXT_DIM};">{text}</span>')
        self._status_lbl.setText(text[:90])

    def _on_finished(self, ok: bool, msg: str):
        self._pulse.stop()
        if ok:
            self._status_lbl.setText("✓  Complete")
            self._build_stem_grid(self._stems, self._out_dir, self._song_name)
            QTimer.singleShot(500, lambda: self._nav_to(2))
        else:
            self._status_lbl.setText(f"✗  {msg}")
            self._log.append(
                f'<span style="color:{ERROR_C};">FAILED: {msg}</span>')

    def _cancel(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        self._pulse.stop()
        self._nav_to(0)

    def _build_stem_grid(self, stems: dict, out_dir: Path, song_name: str):
        while self._song_card_lay.count():
            item = self._song_card_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        n_stems = sum(1 for v in stems.values() if v)
        sc = SongCard(song_name, str(out_dir), n_stems)
        sc.clicked.connect(lambda: self._nav_to(3))
        self._song_card_lay.addWidget(sc)

        while self._stems_list_lay.count() > 1:
            item = self._stems_list_lay.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        self._stems_list_title.setText(song_name.upper())
        self._stems_list_path.setText(str(out_dir))
        for name in ["vocals", "instrumental", "bass", "drums", "other"]:
            path = stems.get(name)
            if path:
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
        self._nav_to(4)

    def _open_detail(self, stem_name: str, file_path: str):
        self._open_player(stem_name, file_path)

    def _stop_players(self):
        for p in self._active_players:
            try:
                p.stop_playback()
            except Exception:
                pass
        self._active_players = []

    def _new_separation(self): self._stop_players(); self._nav_to(0)

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
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
        super().closeEvent(e)


_TC = "\033[96m"   
_GRAY = "\033[90m"
_WHT = "\033[97m"
_DIM = "\033[2m"
_RST = "\033[0m"
_W = 62
_DIV = f"{_TC}{'─' * _W}{_RST}"


def _tui_header(text: str):
    print(f"\n{_DIV}\n  {text}\n{_DIV}")


def _tui_step(text: str):
    print(f"\n  {_TC}▸{_RST} {text}")


def _tui_info(text: str):
    print(f"    {_GRAY}[i]{_RST} {text}")


def _tui_warn(text: str):
    print(f"    {_WHT}[⚠]{_RST}  {text}")


def _tui_done(text: str):
    print(f"    {_TC}[✓]{_RST}  {text}")


def _tui_opt(number: str, label: str):
    print(f"  {_TC}[{_GRAY}{number}{_TC}]{_RST}  {_GRAY}{label}{_RST}")


def _tui_prompt_max_options() -> MaxModeOptions:
    print(f"\n{_DIV}\n  MAX MODE\n{_DIV}")

    print(f"\n  {_GRAY}Vocal separation model:{_RST}\n")
    model_keys = list(MDX_MODELS.keys())
    for i, key in enumerate(model_keys, 1):
        m = MDX_MODELS[key]
        print(f"  {_TC}[{_GRAY}{i}{_TC}]{_RST}  {_GRAY}{m['display']}{_RST}")
    while True:
        raw = input(f"\n  {_TC}>{_RST} ").strip() or "1"
        if raw.isdigit() and 1 <= int(raw) <= len(model_keys):
            mdx_key = model_keys[int(raw) - 1]
            break
        print(f"  {_GRAY}Invalid choice.{_RST}")
    _tui_done(MDX_MODELS[mdx_key]['display'])

    print(f"\n{_DIV}\n  {_GRAY}Post-processing{_RST}\n")
    dr_keys = list(DEREVERB_MODELS.keys())
    named_dr = {k: v for k, v in DEREVERB_MODELS.items() if k != "none"}
    named_keys = list(named_dr.keys())
    print(f"  {_TC}[{_GRAY}0{_TC}]{_RST}  {_GRAY}None{_RST}")
    for i, key in enumerate(named_keys, 1):
        m = DEREVERB_MODELS[key]
        print(f"  {_TC}[{_GRAY}{i}{_TC}]{_RST}  {_GRAY}{m['display']}{_RST}")
    while True:
        raw = input(f"\n  {_TC}>{_RST} ").strip() or "2"
        if raw == "0":
            dr_key = None
            break
        if raw.isdigit() and 1 <= int(raw) <= len(named_keys):
            dr_key = named_keys[int(raw) - 1]
            break
        print(f"  {_GRAY}Invalid choice.{_RST}")
    _tui_done(DEREVERB_MODELS[dr_key]['display']
              if dr_key else "Post-processing disabled")

    print(f"\n{_DIV}\n  {_GRAY}De-bleed{_RST}\n")
    print(
        f"  {_DIM}Attenuates rhythmic bleed (snare/hi-hat) that leaked through.{_RST}\n")
    debleed = input(
        f"  Enable de-bleed? {_GRAY}[y/N]{_RST}: ").strip().lower() == 'y'
    _tui_done(f"De-bleed {'enabled' if debleed else 'disabled'}")

    return MaxModeOptions(mdx_model_key=mdx_key, dereverb_model_key=dr_key, debleed=debleed)


def _tui_check_dependencies() -> bool:
    if sys.version_info < (3, 9):
        print(f"Python 3.9+ required.  Current: {sys.version}")
        return False

    missing = []
    for pkg, label in [
        ("torch",           "torch"),
        ("demucs",          "demucs"),
        ("audio_separator", "audio-separator"),
        ("soundfile",       "soundfile"),
        ("numpy",           "numpy"),
        ("onnxruntime",     "onnxruntime"),
    ]:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(label)

    if missing:
        print(f"Missing: {', '.join(missing)}")
        print("  pip install demucs audio-separator onnxruntime torch soundfile numpy")
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
        _tui_info("NVIDIA  pip install onnxruntime-gpu")
        _tui_info(
            "AMD     pip install torch --index-url https://download.pytorch.org/whl/rocm5.7")
        _tui_info("Intel   pip install openvino onnxruntime-openvino")

    vram = _get_vram_free_mb()
    ram = _get_ram_free_mb()
    if vram > 0:
        _tui_info(f"VRAM free: {vram:.0f} MB")
    _tui_info(f"RAM free:  {ram:.0f} MB")

    print(_DIV)
    return True


def _tui_run():
    backend, backend_desc = HardwareDetector.get_best_backend()
    onnx_providers = HardwareDetector.get_onnx_providers(backend)

    os.system('cls' if os.name == 'nt' else 'clear')

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
    print()
    print(_DIV)
    print()
    _tui_opt("1", "Fast             — htdemucs · 1 shift · quick results")
    _tui_opt("2", "Fine-tuned       — htdemucs_ft · resource-aware shifts")
    _tui_opt("3", "Professional     — MDX + Demucs ensemble")
    _tui_opt("4", "Max              — full ensemble + post-processing")
    print()

    while True:
        choice = input(f"  {_TC}>{_RST} ").strip()
        if choice.lower() == 'q':
            print(f"\n  {_GRAY}Goodbye!{_RST}\n")
            sys.exit(0)
        if choice in ('1', '2', '3', '4'):
            break
        print(f"  {_GRAY}Invalid — enter 1–4 or q to quit.{_RST}")

    preset = {'1': 'fast', '2': 'fine-tuned',
              '3': 'professional', '4': 'max'}[choice]
    max_options = _tui_prompt_max_options() if preset == "max" else None

    print(f"\n{_DIV}\n  {_GRAY}Export format{_RST}\n")
    _tui_opt("1", "WAV  (lossless · 24-bit)  [default]")
    _tui_opt("2", "MP3  (320 kbps)")
    fmt_raw = input(f"\n  {_TC}>{_RST} ").strip() or "1"
    export_fmt = "mp3" if fmt_raw == "2" else "wav"
    _tui_done(f"Export format: {export_fmt.upper()}")

    print(f"\n{_DIV}")
    while True:
        raw = input(f"  {_GRAY}Audio file path:{_RST} ").strip().strip(
            '"').strip("'")
        audio_path = Path(raw).expanduser()
        if audio_path.exists():
            break
        print(f"  {_GRAY}File not found: {audio_path}{_RST}")

    default_out = Path.cwd() / audio_path.stem
    raw_out = input(
        f"  {_GRAY}Output directory [{default_out}]:{_RST} ").strip()
    output_dir = Path(raw_out).expanduser() if raw_out else default_out

    print(f"\n{_DIV}\n  {_GRAY}Configuration{_RST}\n{_DIV}")
    print(f"  {_GRAY}Preset    {_RST}{preset.upper()}")
    print(f"  {_GRAY}Backend   {_RST}{backend_desc}")
    print(f"  {_GRAY}Format    {_RST}{export_fmt.upper()}")
    print(f"  {_GRAY}Input     {_RST}{audio_path}")
    print(f"  {_GRAY}Output    {_RST}{output_dir}")
    if max_options:
        print(
            f"  {_GRAY}MDX       {_RST}{MDX_MODELS[max_options.mdx_model_key]['display']}")
        dr = max_options.dereverb_model_key
        print(
            f"  {_GRAY}Post      {_RST}{DEREVERB_MODELS[dr]['display'] if dr else 'disabled'}")
        print(
            f"  {_GRAY}De-bleed  {_RST}{'enabled' if max_options.debleed else 'disabled'}")

    gpu = backend in (HardwareBackend.CUDA,
                      HardwareBackend.ROCM, HardwareBackend.TENSORRT)
    gs, cs = _resource_aware_shifts(preset, gpu)
    print(f"  {_GRAY}Shifts    {_RST}{gs if gpu else cs} ")
    print(_DIV)

    if input(f"\n  Proceed? {_GRAY}[Y/n]{_RST}: ").strip().lower() not in ('', 'y'):
        print(f"\n  {_GRAY}Cancelled.{_RST}\n")
        sys.exit(0)

    cfg = SeparationConfig.from_preset(preset, backend, max_options=max_options,
                                       export_format=export_fmt)
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

    argv = [a for a in sys.argv if a != "--tui"]
    app = QApplication(argv + ["-style", "Fusion"])
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


if __name__ == "__main__":
    main()
