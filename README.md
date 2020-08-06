# Python Speech Features CUDA

This package is a [Python Speech Features](https://github.com/jameslyons/python_speech_features) reimplementation that offers up to hundreds of times speedup on CUDA enabled GPUs. The API is designed to be as close as possible to the original implementation such that users may have their exsiting projects benefitted from the speedup with least modifications to the code.

## Get Started

This section will walk you through the installation and prerequisites.

### Dependencies

The package was developped on the following dependencies:

1. NumPy (1.19 or greater).
2. CuPy (7.6 or greater).

Please note that the dependencies may require Python 3.7 or greater. It is recommended to install and maintain all packages using [`conda`](https://www.anaconda.com/) or [`pip`](https://pypi.org/project/pip/). To install CuPy, additional effort is needed to get CUDA mounted. Please check the official websites of [CUDA](https://developer.nvidia.com/cuda-downloads) for detailed instructions. Also, since this pakage only uses the most generic functions that are expected to be invariant through dependencies' versions, it will possibly be working well even with lower versions.

### Installation

To install from [PyPI](https://pypi.org/project/python-speech-features-cuda/):

```
pip install python_speech_features_cuda
```

To install from GitHub repo using `pip`:

```
pip install git+git://github.com/vkola-lab/python_speech_features_cuda
```

## What Is Different

