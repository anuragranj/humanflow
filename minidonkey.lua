--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'

local M = {}

local eps = 1e-6
local meanstdCache = 'models/meanstdCache.t7'
local meanstd = torch.load(meanstdCache)
local mean = meanstd.mean
local std = meanstd.std

local  modelL0, modelL1, modelL2, modelL3, modelL4

if opt.level > 0 then
  local modelL0path = paths.concat('models', 'model_00.t7')
  modelL0 = torch.load(modelL0path)
  if torch.type(modelL0) == 'nn.DataParallelTable' then
    modelL0 = modelL0:get(1)
  end
  modelL0:evaluate()
end


if opt.level > 1 then
   -- Load modelL1
   local modelL1path = paths.concat('models', 'modelL1_2.t7')
   modelL1 = torch.load(modelL1path)
   if torch.type(modelL1) == 'nn.DataParallelTable' then
      modelL1 = modelL1:get(1)
   end
   modelL1 = cudnn.convert(modelL1, nn):float()
   modelL1:evaluate()
end

if opt.level > 2 then
-- Load modelL2
   local modelL2path = paths.concat('models', 'modelL2_2.t7')
   modelL2 = torch.load(modelL2path)
   if torch.type(modelL2) == 'nn.DataParallelTable' then
      modelL2 = modelL2:get(1)
   end
   modelL2 = cudnn.convert(modelL2, nn):float()
   modelL2:evaluate()
end

if opt.level > 3 then
   -- Load modelL3
   local modelL3path = paths.concat('models', 'modelL3_2.t7')
   modelL3 = torch.load(modelL3path)
   if torch.type(modelL3) == 'nn.DataParallelTable' then
      modelL3 = modelL3:get(1)
   end
   modelL3 = cudnn.convert(modelL3, nn):float()
   modelL3:evaluate()
end

if opt.level > 4 then
   -- Load modelL4
   local modelL4path = paths.concat('models', 'modelL4_1.t7')
   modelL4 = torch.load(modelL4path)
   if torch.type(modelL4) == 'nn.DataParallelTable' then
      modelL4 = modelL4:get(1)
   end
   modelL4 = cudnn.convert(modelL4, nn):float()
   modelL4:evaluate()
end

local function getTrainValidationSplits(path)
   local numSamples = sys.fexecute( "ls " .. opt.data .. "| wc -l")/3
   --local numSamples = 512
   --print("WARNING: Using only " ..numSamples .." data points")
   local ff = torch.DiskFile(path, 'r')
   local trainValidationSamples = torch.IntTensor(numSamples)
   ff:readInt(trainValidationSamples:storage())
   ff:close()

   local train_samples = trainValidationSamples:eq(1):nonzero()
   local validation_samples = trainValidationSamples:eq(2):nonzero()

   return train_samples, validation_samples
  -- body
end
M.getTrainValidationSplits = getTrainValidationSplits

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   -- find the smaller dimension, and resize it to loadSize (while keeping aspect ratio)
   return input
end
M.loadImage = loadImage

local function loadFlow(filename)
  TAG_FLOAT = 202021.25 
  local ff = torch.DiskFile(filename):binary()
  local tag = ff:readFloat()
  if tag ~= TAG_FLOAT then
    xerror('unable to read '..filename..
     ' perhaps bigendian error','readflo()')
  end
   
  local w = ff:readInt()
  local h = ff:readInt()
  local nbands = 2
  local tf = torch.FloatTensor(h, w, nbands)
  ff:readFloat(tf:storage())
  ff:close()

  local flow = tf:permute(3,1,2)  
  return flow
end
M.loadFlow = loadFlow

local  function rotateFlow(flow, angle)
  local flow_rot = image.rotate(flow, angle)
  local fu = torch.mul(flow_rot[1], math.cos(-angle)) - torch.mul(flow_rot[2], math.sin(-angle)) 
  local fv = torch.mul(flow_rot[1], math.sin(-angle)) + torch.mul(flow_rot[2], math.cos(-angle))
  flow_rot[1]:copy(fu)
  flow_rot[2]:copy(fv)

  return flow_rot
end
M.rotateFlow = rotateFlow

local function scaleFlow(flow, height, width)
  -- scale the original flow to a flow of size height x width
  local sc = height/flow:size(2)
--  print('scale-' .. sc .. ' height-' ..height ..'width-' ..width )
--  print((width/flow:size(3) - sc))
  assert(torch.abs(width/flow:size(3) - sc)<eps, 'Aspect ratio of output flow is not the same as input flow' )
  --local flow_scaled = image.scale(flow, width, height, 'simple')*sc
  
  local flow_scaled = image.scale(flow, width, height)*sc

  return flow_scaled

end
M.scaleFlow = scaleFlow
--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]
local  function computeInitFlowL0(imagesL0)
  local h = 12
  local w = 16

  local  images_scaled = image.scale(imagesL0, w, h)
  local  _flowappend = torch.zeros(2, h, w)
  local  images_in = torch.cat(images_scaled, _flowappend:float(), 1)
  images_in:resize(1,8,h,w)

  local flow_est = modelL0:forward(images_in:cuda())
  return flow_est:squeeze():float()
  -- body
end

local function computeInitFlowL1(imagesL1)
  local h = 24
  local w = 32

  local images_scaled = image.scale(imagesL1, w, h)
  local _flowappend = torch.zeros(2, h, w)
  local images_in = torch.cat(images_scaled, _flowappend:float(), 1)
  images_in:resize(1,8,h,w)

  local flow_est = modelL1:forward(images_in:float())
  return flow_est:squeeze():float()
  -- body
end

local function computeInitFlowL2(imagesL2)
  local h = 48
  local w = 64

  local images_scaled = image.scale(imagesL2, w, h)
  local _flowappend = scaleFlow(computeInitFlowL1(imagesL2), h, w)
  if opt.warp == 1 then
    local _img2 = images_scaled[{{4,6},{},{}}]
    images_scaled[{{4,6},{},{}}]:copy(image.warp(_img2, _flowappend:index(1, torch.LongTensor{2,1})))
  end

  local images_in = torch.cat(images_scaled, _flowappend:float(), 1)
  images_in:resize(1,8,h,w)

  local  flow_est = modelL2:forward(images_in:float())
  return flow_est:squeeze():float():add(_flowappend)
  -- body
end

local function computeInitFlowL3(imagesL3)
  local h = 96
  local w = 128
  local images_scaled = image.scale(imagesL3, w, h)
  local _flowappend = scaleFlow(computeInitFlowL2(imagesL3), h, w)
  if opt.warp == 1 then
    local _img2 = images_scaled[{{4,6},{},{}}]
    images_scaled[{{4,6},{},{}}]:copy(image.warp(_img2, _flowappend:index(1, torch.LongTensor{2,1})))
  end

  local images_in = torch.cat(images_scaled, _flowappend:float(), 1)
  images_in:resize(1,8,h,w)

  local flow_est = modelL3:forward(images_in:float())
  return flow_est:squeeze():float():add(_flowappend)
  -- body
end

local  function computeInitFlowL4(imagesL4)
  local h = 192
  local w = 256
  local images_scaled = image.scale(imagesL4, w, h)
  local _flowappend = scaleFlow(computeInitFlowL3(imagesL4), h, w)
  if opt.warp == 1 then
    local _img2 = images_scaled[{{4,6},{},{}}]
    images_scaled[{{4,6},{},{}}]:copy(image.warp(_img2, _flowappend:index(1, torch.LongTensor{2,1})))
  end

  local images_in = torch.cat(images_scaled, _flowappend:float(), 1)
  images_in:resize(1,8,h,w)

  local flow_est = modelL4:forward(images_in:float())
  return flow_est:squeeze():float():add(_flowappend) 
  -- body
end

local function makeData(images, flows)
  local initFlow, flowDiffOutput
  local images_scaled = image.scale(images, opt.fineWidth, opt.fineHeight)
  
  if opt.level == 0 then
    initFlow = torch.zeros(2, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth) 

  elseif opt.level == 1 then
    initFlow = torch.zeros(2, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)

  elseif opt.level == 2 then
    assert(opt.fineWidth==64, 'Level width mismatch')
    initFlow = computeInitFlowL1(images)
    initFlow = scaleFlow(initFlow, opt.fineHeight, opt.fineWidth)

    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = flowDiffOutput:add(flowDiffOutput, -1, initFlow)

  elseif opt.level == 3 then
    assert(opt.fineWidth==128, 'Level width mismatch')
    initFlow = computeInitFlowL2(images)
    initFlow = scaleFlow(initFlow, opt.fineHeight, opt.fineWidth)

    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = flowDiffOutput:add(flowDiffOutput, -1, initFlow)

  elseif opt.level == 4 then
    assert(opt.fineWidth == 256, 'Level width mismatch')
    initFlow = computeInitFlowL3(images)
    initFlow = scaleFlow(initFlow, opt.fineHeight, opt.fineWidth)

    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = flowDiffOutput:add(flowDiffOutput, -1, initFlow)

  elseif opt.level == 5 then
    assert(opt.fineWidth == 512, 'Level width mismatch')
    initFlow = computeInitFlowL4(images)
    initFlow = scaleFlow(initFlow, opt.fineHeight, opt.fineWidth)
    
    flowDiffOutput = scaleFlow(flows, opt.fineHeight, opt.fineWidth)
    flowDiffOutput = flowDiffOutput:add(flowDiffOutput, -1, initFlow)

  end

  if opt.warp == 1 then
    local _img2 = images_scaled[{{4,6},{},{}}]:clone()
    images_scaled[{{4,6},{},{}}]:copy(image.warp(_img2, initFlow:index(1, torch.LongTensor{2,1})))
  end

  if opt.polluteFlow == 1 then
    initFlow = initFlow + torch.rand(initFlow:size()):mul(2):csub(1)
  end

  local imageFlowInputs = torch.cat(images_scaled, initFlow:float(), 1)

  --print('Printing makeData')
  --print(imageFlowInputs:size())
  --print(flowDiffOutput:size())

  return imageFlowInputs, flowDiffOutput
end

M.makeData = makeData

local function getRawData(id)
   local path1 = paths.concat(opt.data, (string.format("%05i", id) .."_img1.ppm"))
   local path2 = paths.concat(opt.data, (string.format("%05i", id) .."_img2.ppm"))
   
   local img1 = loadImage(path1)
   local img2 = loadImage(path2)
   
   local pathF = paths.concat(opt.data, (string.format("%05i", id) .."_flow.flo"))
   local flow = loadFlow(pathF)

   return img1, img2, flow
end
M.getRawData = getRawData

local testHook = function(id)
   local path1 = paths.concat(opt.data, (string.format("%05i", id) .."_img1.ppm"))
   local path2 = paths.concat(opt.data, (string.format("%05i", id) .."_img2.ppm"))
   
   local img1 = loadImage(path1)
   local img2 = loadImage(path2)
   local images = torch.cat(img1, img2, 1)
   
   local pathF = paths.concat(opt.data, (string.format("%05i", id) .."_flow.flo"))
   local flow = loadFlow(pathF)
   
   for i=1,6 do -- channels
      if mean then images[{{i},{},{}}]:add(-mean[i]) end
      if std then images[{{i},{},{}}]:div(std[i]) end
   end


   return makeData(images, flow)
end
M.testHook = testHook

local function writeFlow(filename, F)
  F = F:permute(2,3,1)
  TAG_FLOAT = 202021.25 
  local ff = torch.DiskFile(filename, 'w'):binary()
  ff:writeFloat(TAG_FLOAT)
   
  ff:writeInt(F:size(2)) -- width
  ff:writeInt(F:size(1)) -- height

  ff:writeFloat(F:storage())
  ff:close()
end
M.writeFlow = writeFlow

local saveHook = function(id, flow, saveDir)
   local pathF = paths.concat(saveDir, (string.format("%05i", id) .."_flow.flo"))
   print('Saving to ' ..pathF)
   writeFlow(pathF, flow)

end
M.saveHook = saveHook

---------------
-- FLOW UTILS
---------------
local function computeNorm(...)
   -- check args
   local _, flow_x, flow_y = xlua.unpack(
      {...},
      'opticalflow.computeNorm',
      'computes norm (size) of flow field from flow_x and flow_y,\n',
      {arg='flow_x', type='torch.Tensor', help='flow field (x), (WxH)', req=true},
      {arg='flow_y', type='torch.Tensor', help='flow field (y), (WxH)', req=true}
   )
   local flow_norm = torch.Tensor()
   local x_squared = torch.Tensor():resizeAs(flow_x):copy(flow_x):cmul(flow_x)
   flow_norm:resizeAs(flow_y):copy(flow_y):cmul(flow_y):add(x_squared):sqrt()
   return flow_norm
end
M.computeNorm = computeNorm
------------------------------------------------------------
-- computes angle (direction) of flow field from flow_x and flow_y,
--
-- @usage opticalflow.computeAngle() -- prints online help
--
-- @param flow_x  flow field (x), (WxH) [required] [type = torch.Tensor]
-- @param flow_y  flow field (y), (WxH) [required] [type = torch.Tensor]
------------------------------------------------------------
local function computeAngle(...)
   -- check args
   local _, flow_x, flow_y = xlua.unpack(
      {...},
      'opticalflow.computeAngle',
      'computes angle (direction) of flow field from flow_x and flow_y,\n',
      {arg='flow_x', type='torch.Tensor', help='flow field (x), (WxH)', req=true},
      {arg='flow_y', type='torch.Tensor', help='flow field (y), (WxH)', req=true}
   )
   local flow_angle = torch.Tensor()
   flow_angle:resizeAs(flow_y):copy(flow_y):cdiv(flow_x):abs():atan():mul(180/math.pi)
   flow_angle:map2(flow_x, flow_y, function(h,x,y)
				      if x == 0 and y >= 0 then
					 return 90
				      elseif x == 0 and y <= 0 then
					 return 270
				      elseif x >= 0 and y >= 0 then
					 -- all good
				      elseif x >= 0 and y < 0 then
					 return 360 - h
				      elseif x < 0 and y >= 0 then
					 return 180 - h
				      elseif x < 0 and y < 0 then
					 return 180 + h
				      end
				   end)
   return flow_angle
end
M.computeAngle = computeAngle
------------------------------------------------------------
-- merges Norm and Angle flow fields into a single RGB image,
-- where saturation=intensity, and hue=direction
--
-- @usage opticalflow.field2rgb() -- prints online help
--
-- @param norm  flow field (norm), (WxH) [required] [type = torch.Tensor]
-- @param angle  flow field (angle), (WxH) [required] [type = torch.Tensor]
-- @param max  if not provided, norm:max() is used [type = number]
-- @param legend  prints a legend on the image [type = boolean]
------------------------------------------------------------
local function field2rgb(...)
   -- check args
   local _, norm, angle, max, legend = xlua.unpack(
      {...},
      'opticalflow.field2rgb',
      'merges Norm and Angle flow fields into a single RGB image,\n'
	 .. 'where saturation=intensity, and hue=direction',
      {arg='norm', type='torch.Tensor', help='flow field (norm), (WxH)', req=true},
      {arg='angle', type='torch.Tensor', help='flow field (angle), (WxH)', req=true},
      {arg='max', type='number', help='if not provided, norm:max() is used'},
      {arg='legend', type='boolean', help='prints a legend on the image', default=false}
   )
   
   -- max
   local saturate = false
   if max then saturate = true end
   max = math.max(max or norm:max(), 1e-2)
   
   -- merge them into an HSL image
   local hsl = torch.Tensor(3,norm:size(1), norm:size(2))
   -- hue = angle:
   hsl:select(1,1):copy(angle):div(360)
   -- saturation = normalized intensity:
   hsl:select(1,2):copy(norm):div(max)
   if saturate then hsl:select(1,2):tanh() end
   -- light varies inversely from saturation (null flow = white):
   hsl:select(1,3):copy(hsl:select(1,2)):mul(-0.5):add(1)
   
   -- convert HSL to RGB
   local rgb = image.hsl2rgb(hsl)
   
   -- legend
   if legend then
      _legend_ = _legend_
	 or image.load(paths.concat(paths.install_lua_path, 'opticalflow/legend.png'),3)
      legend = torch.Tensor(3,hsl:size(2)/8, hsl:size(2)/8)
      image.scale(_legend_, legend, 'bilinear')
      rgb:narrow(1,1,legend:size(2)):narrow(2,hsl:size(2)-legend:size(2)+1,legend:size(2)):copy(legend)
   end
   
   -- done
   return rgb
end
M.field2rgb = field2rgb
------------------------------------------------------------
-- Simplifies display of flow field in HSV colorspace when the
-- available field is in x,y displacement
--
-- @usage opticalflow.xy2rgb() -- prints online help
--
-- @param x  flow field (x), (WxH) [required] [type = torch.Tensor]
-- @param y  flow field (y), (WxH) [required] [type = torch.Tensor]
------------------------------------------------------------
local function xy2rgb(...)
   -- check args
   local _, x, y, max = xlua.unpack(
      {...},
      'opticalflow.xy2rgb',
      'merges x and y flow fields into a single RGB image,\n'
	 .. 'where saturation=intensity, and hue=direction',
      {arg='x', type='torch.Tensor', help='flow field (norm), (WxH)', req=true},
      {arg='y', type='torch.Tensor', help='flow field (angle), (WxH)', req=true},
      {arg='max', type='number', help='if not provided, norm:max() is used'}
   )
   
   local norm = computeNorm(x,y)
   local angle = computeAngle(x,y)
   return field2rgb(norm,angle,max)
end
M.xy2rgb = xy2rgb

-----------------
-----------------
return M


