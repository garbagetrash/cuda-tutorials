
CUDA Tutorials
==============

A collection of quick CUDA tutorials to get me started with GPGPU computing.

Covers [the basics](https://devblogs.nvidia.com/even-easier-introduction-cuda/)
and
[streams](https://devblogs.nvidia.com/gpu-pro-tip-cuda-7-streams-simplify-concurrency/).

Also useful are the documentation pages for
[cufft](https://docs.nvidia.com/cuda/cufft/index.html), the [CUDA C Best
Practices
Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html),
and the [CUDA C Programming
Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html).


Installation
------------

Remove all your current Nvidia stuff with

```sh
$ sudo apt -y remove nvidia-*
```

Follow the instructions [here](https://developer.nvidia.com/cuda-downloads) to
get the CUDA Toolkit 10.1 Update 2 installed.

Profilers
---------

The profilers have to be run as root due to some side channel attack possible
through them.  Add the following to your `.bashrc`:

```
alias sudo='sudo '
alias nvprof='/usr/local/cuda-10.1/bin/nvprof'
alias nvvp='/usr/local/cuda-10.1/bin/nvvp'
```

And run `nvvp` with the `-vm` option set so it knows which Java installation to
use:

```sh
$ sudo nvvp -vm /usr/lib/jvm/java-8-openjdk-amd64/jre/bin/java ./stream_legacy
```
