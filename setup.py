import subprocess
import sys
import os

VENV_DIR = ".venv"
HOME = os.path.dirname(os.path.abspath(__file__))

venv_python = os.path.join(HOME, VENV_DIR, "Scripts", "python.exe") if sys.platform == "win32" else os.path.join(VENV_DIR, "bin", "python")

os.system('py -m pip install --user virtualenv')
os.system('py -3.13 -m venv .venv')

os.system(f'{venv_python} -m pip install --upgrade pip')
os.system(f'{venv_python} -m pip install -r requirements.txt') # --index-url https://download.pytorch.org/whl/cu128 - it is for CUDA 12.8 for RTX 5070 Ti, but not compatible with python 3.8