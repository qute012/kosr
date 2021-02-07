# Korean Online Speech Recognition

Implement [Transformer Transducer]. This repository provides end-to-end training 1,000 hours KsponSpeech dataset.
KsponSpeech dataset was processed by referring to [here].

## Preparation
You can download dataset at [AI-Hub]. And the structure of the directory should be prepared for getting started as shown below. Preprocesses were used [ESPnet] for normalizing text  from KsponSpeech recipe. It is provided simply as .trn extention files.
```
root
└─ KsponSpeech_01
└─ KsponSpeech_02
└─ KsponSpeech_03
└─ KsponSpeech_04
└─ KsponSpeech_05
└─ KsponSpeech_eval
└─ scripts
```

## Environment

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
RNN-transducer loss can be installed if the installation is completed.

python>=3.6 & pytorch >= 1.7.0 & torchaudio >= 0.7.0

```
pip install torch==1.7.0+cu101 torchaudio==0.7.0 -f https://download.pytorch.org/whl/torch_stable.html
```

## Usage
Before training, you should already get Ai-Hub dataset. And you needs to check configuration in conf directory and set batch size as fittable as your gpu environment. If you want to use custom configuration, use conf option(default: config/ksponspeech_transducer_base.yaml).
```
python train.py [--conf config-path]
```
after training, create checkpoint directory automatically. You can check saved model at checkpoint directory.
If you want to train continuosly, use continue_from option.
```
python train.py --conf model-configuration --continue_from saved-model-path
```

## Results
|Epoch|Model|CER|WER|
|-----|------|---|---|
|2|Transformer|26%|45%|

[Transformer Transducer]:https://arxiv.org/pdf/2002.02562.pdf
[here]:https://www.mdpi.com/2076-3417/10/19/6936
[AI-Hub]: https://www.aihub.or.kr/aidata/105
[ESPnet]: https://github.com/espnet/espnet/tree/master/egs/ksponspeech/asr1

## Author
Email: 406023@naver.com
