-- Copyright 2016 Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided for research purposes only.  
-- By using this software you agree to the terms of the license file 
-- in the root folder.
-- For commercial use, please contact ps-license@tue.mpg.de.

require 'image'
local TF = require 'transforms'
require 'cutorch'
require 'nn'
require 'cunn'
require 'cudnn'
require 'nngraph'
require 'stn'
require 'spy'
local stringx = require 'pl.stringx'
local flowX = require 'flowExtensions'

local M = {}

local eps = 1e-6
local meanstd = {
   mean = { 0.485, 0.456, 0.406 },
   std = { 0.229, 0.224, 0.225 },
}
local pca = {
   eigval = torch.Tensor{ 0.2175, 0.0188, 0.0045 },
   eigvec = torch.Tensor{
      { -0.5675,  0.7192,  0.4009 },
      { -0.5808, -0.0045, -0.8140 },
      { -0.5836, -0.6948,  0.4203 },
   },
}

local mean = meanstd.mean
local std = meanstd.std
------------------------------------------
local model
local modelpath

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   return input
end
M.loadImage = loadImage

local function getTrainValidationSplits(valFile)
   local valfile = torch.DiskFile(valFile)
   local valdata = valfile:readString("*a")
   local validation_samples = stringx.split(valdata, "\n")
   validation_samples:pop()
   valfile:close()
   return validation_samples
end
M.getTrainValidationSplits = getTrainValidationSplits

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

  local flow = tf:permute(3,1,2):clone()  
  return flow
end
M.loadFlow = loadFlow

local function DeAdjustFlow(flow, h, w)
  local sc_h = h/flow:size(2)
  local sc_w = w/flow:size(3)
  flow = image.scale(flow, w, h, 'simple')
  flow[2] = flow[2]*sc_h
  flow[1] = flow[1]*sc_w

  return flow
end
M.DeAdjustFlow = DeAdjustFlow

local function normalize(imgs)
  return TF.ColorNormalize(meanstd)(imgs)
end
M.normalize = normalize

local easyComputeFlow = function(im1, im2)
  local imgs = torch.cat(im1, im2, 1)
  imgs = TF.ColorNormalize(meanstd)(imgs)


  local width = imgs:size(3)
  local height = imgs:size(2)
  
  local fineWidth, fineHeight
  
  if (width%32 == 0 and width <=512) then
    fineWidth = width
  else
    fineWidth = math.min(512, width - math.fmod(width, 32))
  end

  if (height%32 == 0 and height <=512) then
    fineHeight = height
  else
    fineHeight = math.min(512, height - math.fmod(height, 32))
  end  
       
  imgs = image.scale(imgs, fineWidth, fineHeight)

  imgs = imgs:resize(1,6,imgs:size(2),imgs:size(3)):cuda()
  local flow_est = model:forward(imgs)
  local flow_est_256 = flow_est[4]:squeeze():float()
  
  flow_est_256 = DeAdjustFlow(flow_est_256, height, width)

  return flow_est_256

end

local function easy_setup(modelpath)
  --modelpath = modelpath or paths.concat('models', 'model_best_final.t7')
  print('Loaded model' .. modelpath)
  model = torch.load(modelpath)
  if torch.type(model) == 'nn.DataParallelTable' then
     model = model:get(1)
  end
  model:evaluate()

  return easyComputeFlow
end
M.easy_setup = easy_setup


return M
