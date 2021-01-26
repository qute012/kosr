# Korean-Online-Speech-Recognition

Implement [Transformer Transducer]. This repository provides end-to-end training 1,000 hours KsponSpeech dataset.

# Installation

Warp-transducer needs to install gcc++5 and export CUDA environment variable.

CUDA_HOME settings

```
export CUDA_HOME=$HOME/tools/cuda-9.0 # change to your path
export CUDA_TOOLKIT_ROOT_DIR=$CUDA_HOME
export LD_LIBRARY_PATH="$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
export LIBRARY_PATH=$CUDA_HOME/lib64:$LIBRARY_PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
export CFLAGS="-I$CUDA_HOME/include $CFLAGS"
```

Install gcc++5 and update alternatives

```
sudo add-apt-repository ppa:ubuntu-toolchain-r/test
sudo apt-get update
sudo apt-get install gcc-5 g++-5
sudo update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-5 1
```

python>=3.6 & pytorch >= 1.7.0 & torchaudio >= 0.7.0

```
pip install torch==1.7.0+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

[Transformer Transducer]:https://arxiv.org/pdf/2002.02562.pdf
