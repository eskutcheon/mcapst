

# TODO: I'll probably be moving this and other setup scripts from `mcapst/setup/` to the top-level directory above `mcapst/`



"""
    considering creating some setup utils to be called by setup scripts elsewhere
    - PEP-517 says we can dynamically build dependencies based on user environments
    > "It is also possible for a build backend to provide dynamically calculated build dependencies,
    >   using [**PEP 517**](https://peps.python.org/pep-0517/)'s `get_requires_for_build_wheel` hook.
    > This hook will be called by pip, and dependencies it describes will also be installed in the build environment.
    >   For example, newer versions of setuptools expose the contents of `setup_requires` to pip via this hook."
    >   \- [the pip documentation](https://pip.pypa.io/en/stable/reference/build-system/pyproject-toml/)
"""


import subprocess

def detect_cuda_version():
    try:
        output = subprocess.check_output(['nvcc', '--version']).decode()
        for line in output.split('\n'):
            if 'release' in line:
                return line.split('release')[-1].split(',')[0].strip()
    except Exception: # would normally raise a CalledProcessError if nvcc is not found, but we just consider that "No CUDA"
        return None


def get_cuda_wheel_url(cuda_version: str):
    if cuda_version:
        # should work for torch, torchvision, torchaudio, etc.
        return f"https://download.pytorch.org/whl/cu{cuda_version.replace('.', '')}"
    return "https://download.pytorch.org/whl/cpu"


