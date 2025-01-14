from setuptools import setup, find_packages

setup(
    name="mcapst",
    version="1.0.0",
    #  (cite paper and repo later)
    description="Style transfer for images and videos based on CAP-VSTNet",
    author="eskutcheon",
    packages=find_packages(),
    include_package_data=True,
    # TODO: list actual dependencies later
    install_requires=[
        "torch>=1.9",
        "torchvision>=0.10",
        "numpy",
        "Pillow",
    ],
)
