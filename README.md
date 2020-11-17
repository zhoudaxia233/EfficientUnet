# EfficientUnet
Keras Implementation of Unet with [EfficientNet](https://arxiv.org/abs/1905.11946) as encoder

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
## Special Notice
When I built this, `tensorflow 1.13.1` and `keras 2.2.4` are the latest. There was no `TF2.0`. All the functions and the so-called "best practices" I used in this project may be obsolete. Anyway, this library still works. But please keep in mind, this is built before the advent of `TF2.0`.

---
## Installation
Install `efficientunet`:

```bash
pip install efficientunet
```

---
## Basic Usage

```bash
from efficientunet import *

model = get_efficient_unet_b5((224, 224, 3), pretrained=True, block_type='transpose', concat_input=True)
model.summary()

```

---
## Useful notes
1. This library assumes `channels_last` !
2. You cannot specify `None` for `input_shape`, since the `input_shape` is heavily used in the code for inferring
the architecture. (*The EfficientUnets are constructed dynamically*)
3. Since you cannot use `None` for `input_shape`, the image size for training process and for inference process
have to be the same.  
If you do need to use a different image size for inference, a feasible solution is:
    1. Save the weights of your well-trained model
    2. Create a new model with the desired input shape
    3. Load the weights of your well-trained model into this newly created model

4. Due to some rounding problem in the decoder path (*not a bug, this is a feature* :smirk:), the input shape should be 
divisible by 32.  
e.g. 224x224 is a suitable size for input images, but 225x225 is not.

---
## Potential issues
- If you are unable to load the model from the saved HDF5 file, please refer to 
[this](https://github.com/keras-team/keras/issues/3867) issue.  
- Especially [this](https://github.com/keras-team/keras/issues/3867#issuecomment-313336090) comment can be used as a workaround.

---
## Acknowledgment
0. Some code snippets of EfficientNet are directly borrowed from [this](https://github.com/mingxingtan/efficientnet) repo.
1. The links of pretrained weights are borrowed from [this](https://github.com/qubvel/efficientnet) repo.
