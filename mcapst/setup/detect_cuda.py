

# TODO: I'll probably be moving this and other setup scripts from `mcapst/setup/` to the top-level directory above `mcapst/`

import subprocess

def detect_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode()
        for line in output.split('\n'):
            if 'release' in line:
                return line.split('release')[-1].split(',')[0].strip()
    except Exception: # would normally raise a CalledProcessError if nvcc is not found, but we just consider that "No CUDA"
        return None

def get_cuda_wheel_url(cuda_version):
    if cuda_version:
        # should work for torch, torchvision, torchaudio, etc.
        return f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
    return "https://download.pytorch.org/whl/cpu"


