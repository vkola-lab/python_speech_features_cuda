# Python Speech Features CUDA

This package is a [Python Speech Features](https://github.com/jameslyons/python_speech_features) reimplementation that offers up to hundreds of times performance speedup on CUDA enabled GPUs. The API is designed to be as close as possible to the original implementation such that users may have their existing projects benefited from the acceleration with least modifications to the code.

If you do not have the access to a CUDA GPU, this package may also get you a decent boost (i.e. roughly x2) over the original implementation.

## Get Started

This section will walk you through the installation and prerequisites.

### Dependencies

The package was developed on the following dependencies:

1. NumPy (1.19 or greater).
2. CuPy (7.6 or greater).

Please note that the dependencies may require Python 3.7 or greater. It is recommended to install and maintain all packages using [`conda`](https://www.anaconda.com/) or [`pip`](https://pypi.org/project/pip/). To install CuPy, additional effort is needed to get CUDA mounted. Please check the official websites of [CUDA](https://developer.nvidia.com/cuda-downloads) for detailed instructions. Also, since this package only uses the most generic functions that are expected to be invariant through dependencies' versions, it will possibly be working well even with lower versions.

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

All changes are made around the point of performance gain.

### Intermediate result buffer

Intermediate results (e.g. Mel filterbank and DCT matrix) can be buffered to avoid duplicated computation as long as all parameters remain the same. It is possibly the major reason why this implementation is still faster on CPU.

### Batch process

The original implementation can process only one signal sequence at a time that is definitely a sufficient manner within CPU-only environment. Overly vectorizing NumPy code is actually harmful to the performance in theory due to cache-miss issue. However, GPU is another story that only if you letting it process as many signals as possible at once can unleash its power of parallelism. Here, you can call functions with multiple sequences as a batch ndarray whose preceding dimensions are batch dimensions.
