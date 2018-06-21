# Learning Human Optical Flow
This code is based on the paper [Learning Human Optical Flow](https://arxiv.org/abs/1806.05666).

* [Data:](#data)  Downloading the data
* [Trained Models:](#models)  Downloading the trained models
* [Setup:](#setUp)  Setting up this code
* [Usage:](#usage) Computing Human Optical Flow
* [Training:](#training) Train your own models on mulitiple GPUs
* [References:](#references) For further reading

<a name="data"></a>
## Data
### We are working releasing the training data soon.

<a name="models"></a>
## Data
### We are working releasing the trained models soon.

<a name="setUp"></a>
## Setup
You need to have [Torch.](http://torch.ch/docs/getting-started.html#_)

Install other required packages
```bash
cd extras/spybhwd
luarocks make
cd ../stnbhwd
luarocks make
```
<a name="usage"></a>
## Usage
#### Load the model
```lua
humanflownet = torch.load(<MODEL_PATH>):cuda()
```
#### Load images and compute flow
```lua
im1 = image.load(<IMAGE_PATH_1>)
im2 = image.load(<IMAGE_PATH_2>)
flow = humanflownet:forward(im1, im2)
```

<a name="training"></a>
## Training
```bash
th main.lua -netType fullBodyModel -nGPU 4 -nDonkeys 16 -LR 1e-6 -epochSize 1000 -data <PATH_TO_DATASET> -trainFile <LIST_OF_TRAINING_SAMPLES.txt> -valFile <LIST_OF_VALIDATION_SAMPLES>
```

<a name="references"></a>
## References

1. Training code is based on [anuragranj/spynet.](https://github.com/anuragranj/spynet)
2. Warping code is based on [qassemoquab/stnbhwd.](https://github.com/qassemoquab/stnbhwd)

## License
MIT License, free usage without any warranty. Check LICENSE file for details.

## Citing this code
Ranjan, Anurag, Javier Romero, and Michael J. Black. "Learning Human Optical Flow." arXiv preprint arXiv:1806.05666 (2018).
