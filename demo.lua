bodynet = require 'bodynet'
flowX = require 'flowExtensions'
require 'image'

torch.setdefaulttensortype('torch.FloatTensor')
-- Setup
easyComputeFlow = bodynet.easy_setup('pretrained/human_flow_model_noise_adaptive.t7')

-- Load Images
im1 = image.load('samples/txt_000010.png', 3, 'float')
im2 = image.load('samples/txt_000011.png', 3, 'float')

-- Compute Flow
flow = easyComputeFlow(im1, im2)

-- Visualize Flow by converting it to rgb image
flow_rgb = flowX.xy2rgb(flow[1], flow[2])

-- Write a flow file
flowX.writeFLO('samples/flow.flo', flow)
