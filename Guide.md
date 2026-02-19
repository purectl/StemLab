# StemLab Guide
Install packages

```bash
pip install demucs audio-separator onnxruntime torch soundfile numpy pyqt6
```

If you're on **macos** or **linux** make sure to start a venv. (preferably python3.11)
```bash
python3.11 -m venv stemlab
python -m venv stemlab
```
Note: just run install.sh to **automatically** do this. 

---

## TUI | GUI 
To run GUI just simply run:
```bash
python stemlab.py
```

for tui:
```bash
python stemlab.py --tui
```

---

## Running in colab 
you can run this in **colab** using the `--tui` flag.

For best usage of **colab** use:
- T4 GPU (fast & free)
- if you have **Pro**, use the best one you can for faster separation

---
# Pre-compiled Binaries/exe
**Coming Soon.**
