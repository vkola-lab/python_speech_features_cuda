# Python Speech Features CUDA

This package is a [Python Speech Features](https://github.com/jameslyons/python_speech_features) re-implementation that offers up to hundreds of times performance boost on CUDA enabled GPUs. The API is designed to be as close as possible to the original implementation such that users may have their existing projects benefited from the acceleration with least modifications to the code. If you do not have the access to a CUDA GPU, this package may also get you a decent speedup (i.e. roughly x10) by utilizing multi-core CPU, optimizing RAM usage etc.

![Speedup Plot](/readme_plot/plot.jpg)

The performance of the 3 most important functions, namely `mfcc`, `ssc` and `delta`, were tested on random signals of length 500,000 which are approximately 30 seconds each. Let's take the speed of original implementation as baseline (i.e <img src="https://render.githubusercontent.com/render/math?math=2^0">), the vertical axis tells the speed gain; the horizontal axis signifies the batch size. It is clear to see that the acceleration is universal whichever the backend is NumPy (CPU) or CuPy (CUDA GPU), although the advantage of GPU is way more significant. Please also note the astonishing performance of `delta` function is due to a reworked logic.

Note that the benchmark was run on a system of Intel 8700K (6-core) and NVIDIA GTX 1080Ti, the acutal performance may vary on different settings.

## Get Started

This section will walk us through the installation and prerequisites.

#### Dependencies

The package was developed on the following dependencies:

1. [NumPy](https://numpy.org/) (1.19 or greater).
2. [CuPy](https://cupy.dev/) (7.6 or greater).

Please note that the dependencies may require Python 3.7 or greater. It is recommended to install and maintain all packages using [`conda`](https://www.anaconda.com/) or [`pip`](https://pypi.org/project/pip/). To install CuPy, additional effort is needed to get CUDA mounted. Please check the official websites of [CUDA](https://developer.nvidia.com/cuda-downloads) for detailed instructions. Also, since this package only uses the most generic functions that are expected to be invariant through dependencies' versions, it will possibly be working well even with lower versions.

Optional dependencies:

1. [pyFFTW](https://pypi.org/project/pyFFTW/) (0.12)
2. [Numba](http://numba.pydata.org/) (0.50)

These packages are the powerhouse for CPU based computation. If available, they will be auto-detected and loaded during the initialization stage. Of course You don't need them if you have a CUDA-enabled GPU and go for CuPy as the backend.

#### Installation

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

#### Intermediate result buffer

Intermediate results (e.g. Mel filterbank and DCT matrix) can be buffered to avoid duplicated computation when all parameters remain the same. It is possibly the major reason why this implementation is still faster on CPU.

#### Batch process

The original implementation can process only one signal sequence at a time. Of course, it is a sufficient manner within CPU-only environment, overly vectorizing NumPy code is actually harmful to the performance due to the curse of cache-miss in practice. However, GPU is another story that, roughly speaking, only if we letting it process as many signals as possible at once can unleash its power of parallelism. As we can see from the plot above, GPU code has consistent performance gain as the batch size increases. Here, functions can be fed with multiple sequences as a batch `ndarray` whose preceding dimensions are batch dimensions.

#### Strict floating-point control

Numerical data subtype is almost transparent to Python coders, but it is necessarily explicit for GPU programming. In order to constraint floating-point type, this implementation introduces a global 'knob' indicating what floating-point (i.e. 32 or 64) is expected; any input `ndarray` needs to be consistent with that or a `TypeError` will be raised.

#### API changes

The API is kept almost the same except that sub-module `sigproc` is removed. All functions previously under `sigproc` can now be accessed at the package root level. This is to adopt the 'pythonic' principle of 'flat is better than nested.'

A few function argument names may also be changed to make them appear more unified. For example, `NFFT` and `nfft` are both changed to `nfft`, although we will not notice that if arguments are passed in positional manner.

## Examples

#### Import, then check backend and floating-point type

```python
import python_speech_features_cuda as psf

print(psf.env.backend.__name__)  # >>> cupy
print(psf.env.dtype.__name__)    # >>> float64
```

By default, the backend will be set to CuPy and the data type `float64`. If CuPy is not found in the environment, then the backend will be switched to NumPy automatically at package initialization stage.

#### Change backend and floating-point type

```python
import numpy as np

psf.env.backend = np
psf.env.dtype = np.float32

print(psf.env.backend.__name__)  # >>> numpy
print(psf.env.dtype.__name__)    # >>> float32
```

#### Call MFCC()

```python
# initialize a batch of 4 signals of length 500,000 each
sig = psf.env.backend.random.rand(4, 500000, dtype=psf.env.dtype)

# apply MFCC
fea = psf.mfcc(sig, samplerate=16000, winlen=.025, winstep=.01, numcep=13,
               nfilt=26, nfft=None, lowfreq=0, highfreq=None, preemph=.97,
               ceplifter=22, appendEnergy=True, winfunc=None)

print(fea.shape)  # >>> (4, 3124, 13)
```

Please note that the input array MUST be consistent with the package enviroment in terms of backend and dtype. If our raw data is loaded in different format, use `psf.env.backend.asarray(..., dtype=psf.env.dtype)` for conversion.

#### Call MFCC() with nontrivial window

```python
# calculate window function (vector)
samplerate, winlen = 16000, .025
win_len = int(np.round(samplerate * winlen))
win = psf.env.backend.hamming(win_len).astype(psf.env.dtype)

# apply MFCC
fea = psf.mfcc(sig, nfft=512, winfunc=win)

print(fea.shape)  # >>> (4, 3124, 13)
```

Window function (e.g. `hamming`) has only one degree of freedom that is window/frame length. Since window length doesn't change oftenly in most senarios, it is not necessary to calculate it over and over again at each call. This API change is consistent with the idea of buffering.

#### Interoperability

If we are using CuPy as the backend, then all function outputs are CuPy ndarray stored on GPU memory. Assume the very next stop of the CuPy ndarray is another GPU function but provided by other package/library (e.g. PyTorch, Numba), we can simply pass the memory 'pointer' instead of suffering the huge overhead of GPU->RAM->GPU transfer. Please check this CuPy documentation [page](https://docs.cupy.dev/en/stable/reference/interoperability.html) for details.

## Authors

* **Chonghua Xue**, cxue2@bu.edu - Kolachalama laboratory, Boston University School of Medicine
