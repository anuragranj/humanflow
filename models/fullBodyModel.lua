-- Copyright 2018 Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided without any warranty.
-- By using this software you agree to the terms of the license file
-- in the root folder.

require 'nn'
require 'nngraph'
require 'cunn'
require 'cudnn'
require 'stn'

local modelL1path = paths.concat('models', 'modelL1_F.t7')
local modelL2path = paths.concat('models', 'modelL2_F.t7')
local modelL3path = paths.concat('models', 'modelL3_F.t7')
local modelL4path = paths.concat('models', 'modelL4_F.t7')
--local modelL5path = paths.concat('models', 'modelL5_4.t7')

local modelL1 = torch.load(modelL1path)
if torch.type(modelL1) == 'nn.DataParallelTable' then
   modelL1 = modelL1:get(1)
end

local modelL2 = torch.load(modelL2path)
if torch.type(modelL2) == 'nn.DataParallelTable' then
   modelL2 = modelL2:get(1)
end

local modelL3 = torch.load(modelL3path)
if torch.type(modelL3) == 'nn.DataParallelTable' then
   modelL3 = modelL3:get(1)
end

local modelL4 = torch.load(modelL4path)
if torch.type(modelL4) == 'nn.DataParallelTable' then
   modelL4 = modelL4:get(1)
end

--local modelL5 = torch.load(modelL5path)
--if torch.type(modelL5) == 'nn.DataParallelTable' then
--   modelL5 = modelL5:get(1)
--end


local function createWarpModel()
  local imgData = nn.Identity()()
  local floData = nn.Identity()()

  local img1 = nn.Narrow(2,1,3)(imgData)
  local imgOut = nn.Transpose({2,3},{3,4})(nn.Narrow(2,4,3)(imgData)) -- Warping on the second image
  local floOut = nn.Transpose({2,3},{3,4})(floData)

  local warpImOut = nn.Transpose({3,4},{2,3})(nn.BilinearSamplerBHWD()({imgOut, floOut}))
  local imgs = nn.JoinTable(2)({img1, warpImOut})
  local output = nn.JoinTable(2)({imgs, floData})
  local model = nn.gModule({imgData, floData}, {output})

  return model
end

local function createConvDown()
  local conv = cudnn.SpatialConvolution(6,6,3,3,2,2,1,1)
  conv.weight:fill(0)
  conv.bias:fill(0)
  for i=1,6 do
    conv.weight[i][i]:fill(1/9)
  end
  return conv
--  return nn.SpatialAveragePooling(2,2,2,2)
end

local function createConvUp()
  local convup = cudnn.SpatialFullConvolution(2,2,4,4,2,2,1,1)
  convup.weight:fill(0)
  convup.bias:fill(0)
  for i=1,2 do
    convup.weight[i][i]:fill(1/2)
  end
  return convup
--  return nn.SpatialUpSamplingNearest(2)
end

local warpmodel2 = createWarpModel():cuda()
local warpmodel3 = createWarpModel():cuda()
local warpmodel4 = createWarpModel():cuda()
--local warpmodel5 = createWarpModel():cuda()

--local im5 = nn.Identity()()
local im4 = nn.Identity()()
local im3 = createConvDown()(im4)
local im2 = createConvDown()(im3)
local im1 = createConvDown()(im2)

local _flow1 = modelL1(nn.Padding(2,2)(im1))
local flow1 =  createConvUp()(_flow1)

local _flow2 = nn.CAddTable(1)({flow1, modelL2(warpmodel2({im2, flow1}))})
local flow2 =  createConvUp()(_flow2)

local _flow3 = nn.CAddTable(1)({flow2, modelL3(warpmodel3({im3, flow2}))})
local flow3 =  createConvUp()(_flow3)

local _flow4 = nn.CAddTable(1)({flow3, modelL4(warpmodel4({im4, flow3}))})
--local flow4 =  createConvUp()(_flow4)

--local _flow5 =  nn.CAddTable(1)({flow4, modelL5(warpmodel5({im5, flow4}))})

local model = nn.gModule({im4},{_flow1, _flow2, _flow3, _flow4}):cuda()

if opt.nGPU>0 then
  model:cuda()
  model = makeDataParallel(model, opt.nGPU)
end

return model
