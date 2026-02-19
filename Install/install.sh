#!/bin/bash
python3 -m venv StemLab
source venv/bin/activate
pip install demucs audio-separator onnxruntime torch soundfile numpy pyqt6
