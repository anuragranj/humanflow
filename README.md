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
Download the extract the data.

```bash
for i in `seq -w 1 56`; do wget http://humanflow.is.tuebingen.mpg.de/HumanFlowDataset.7z.0$i -c; done ;
7z x HuamnFlowDataset.7z.001
```
Alternately, you can download from the [webpage](http://humanflow.is.tuebingen.mpg.de/).

<a name="models"></a>
## Trained Models
The pretrained models are available in `pretrained/` directory. There are two models:
1. `human_flow_model.t7` is the original trained model as evaluated in the paper.
2. `human_flow_model_noise_adaptive.t7` is trained with additional noisy data.

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
stn = require 'stn'
bodynet = require 'bodynet'
easyComputeFlow = bodynet.easy_setup('pretrained/human_flow_model_[noise_adaptive].t7')
```
#### Load images and compute flow
```lua
im1 = image.load(<IMAGE_PATH_1>, 3, 'float')
im2 = image.load(<IMAGE_PATH_2>, 3, 'float')
flow = easyComputeFlow(im1, im2)
```
To save or visualize optical flow, refer to `flowExtensions.lua`

<a name="training"></a>
## Training
```bash
th main.lua -netType fullBodyModel -nGPU 4 -nDonkeys 16 -LR 1e-6 -epochSize 1000 -data <PATH_TO_DATASET>
```

<a name="references"></a>
## References

1. Training code is based on [anuragranj/spynet.](https://github.com/anuragranj/spynet)
2. Warping code is based on [qassemoquab/stnbhwd.](https://github.com/qassemoquab/stnbhwd)
3. Additional training data can be found at [gulvarol/surreal.](https://github.com/gulvarol/surreal)

## License
MIT License, free usage without any warranty. Check LICENSE file for details.

## Citing this code
Ranjan, Anurag, Javier Romero, and Michael J. Black. "Learning Human Optical Flow." arXiv preprint arXiv:1806.05666 (2018).
