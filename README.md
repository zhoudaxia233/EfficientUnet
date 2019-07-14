# EfficientUnet
Keras Implementation of Unet with [EfficientNet](https://arxiv.org/abs/1905.11946) as encoder

> Note: This library assumes `channels_last` !!!  
> Note: This library assumes `channels_last` !!!  
> Note: This library assumes `channels_last` !!!
- Unet with EfficientNet encoder
  - EfficientNet-B0
  - EfficientNet-B1
  - EfficientNet-B2
  - EfficientNet-B3
  - EfficientNet-B4
  - EfficientNet-B5
  - EfficientNet-B6
  - EfficientNet-B7
---
## Requirements
1. `tensorflow >= 1.13.1`
2. `Keras >= 2.2.4` (It will automatically be installed when you install `efficientunet`)

---
## Installation
Install `efficientunet`:

```bash
pip install efficientunet
```

---
## Potential issues
- If you are unable to load the model from the saved HDF5 file, please refer to 
[this](https://github.com/keras-team/keras/issues/3867) issue.  
- Especially [this](https://github.com/keras-team/keras/issues/3867#issuecomment-313336090) comment can be used as a workaround.

---
## Acknowledgment
0. Some code snippets of EfficientNet are directly borrowed from [this](https://github.com/mingxingtan/efficientnet) repo.
1. The links of pretrained weights are borrowed from [this](https://github.com/qubvel/efficientnet) repo.
