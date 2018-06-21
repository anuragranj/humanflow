--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'image'
require 'cudnn'
paths.dofile('dataset.lua')
paths.dofile('util.lua')
local flowX = require('flowExtensions')
local stringx = require('pl.stringx')
local TF = require 'transforms' 
print('Donkey Started')
-- This file contains the data-loading logic and details.
-- It is run by each data-loader thread.
------------------------------------------
local eps = 1e-6
-- a cache file of the training metadata (if doesnt exist, will be created)
--local trainCache = paths.concat(opt.cache, 'trainCache.t7')
--local testCache = paths.concat(opt.cache, 'testCache.t7')

-- Mean and Standard deviation
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

-- Check for existence of opt.data
if not os.execute('cd ' .. opt.data) then
    error(("could not chdir to '%s'"):format(opt.data))
end

-- TODO: Get rid of loadSize
local inputSize = opt.inputSize
local flowSize = opt.outputSize

local function loadImage(path)
   local input = image.load(path, 3, 'float')
   return input
end

--local function loadFlow(filename)
--  local u, v, o = png16.readFlowPNG16(filename)
--  local flow = torch.zeros(2, u:size(1), u:size(2))
--  flow[1]:copy(u)
--  flow[2]:copy(v) 

--  return flow, o
--end

local function Preprocess()
  return TF.Compose{
    TF.ColorJitter({
      brightness = 0.4,
      contrast = 0.4,
      saturation = 0.4,
      }),
    TF.Lighting(0.1, pca.eigval, pca.eigvec),
    TF.ColorNormalize(meanstd),
    }
end

local function getTrainValidationSplits(trainFile, valFile)
   local trainfile = torch.DiskFile(trainFile)
   local traindata = trainfile:readString("*a")
   local train_samples = stringx.split(traindata, "\n")
   train_samples:pop()
   trainfile:close()
   local valfile = torch.DiskFile(valFile)
   local valdata = valfile:readString("*a")
   local validation_samples = stringx.split(valdata, "\n")
   validation_samples:pop()
   valfile:close()
   return train_samples, validation_samples
end

train_samples, validation_samples = getTrainValidationSplits(
   paths.concat(opt.data, opt.trainFile), paths.concat(opt.data, opt.valFile))
--------------------------------------------------------------------------------
--[[
   Section 1: Create a train data loader (trainLoader),
   which does class-balanced sampling from the dataset and does a random crop
--]]
-- function to preprocess data for coarse to fine estimation

-- function to load the image, jitter it appropriately (random crops etc.)
local trainHook = function(self, id)
   collectgarbage()
   local pathTable = stringx.split(self.samples[id], " ")
   
   local path1 = paths.concat(opt.data, pathTable[1])
   local path2 = paths.concat(opt.data, pathTable[2])
   local pathF = paths.concat(opt.data, pathTable[3])
   
   local img1 = loadImage(path1)
   local img2 = loadImage(path2)
   local flow = flowX.loadFLO(pathF)

   local images = torch.cat(img1, img2, 1)  
   
   local gH = images:size(2)
   local gW = images:size(3)

   -- Rotation [-0.3 0.3] in radians
   local ang = torch.uniform()*0.6 - 0.3

   -- zoom before rotation
   local cW = gW*math.cos(math.abs(ang)) + gH*math.sin(math.abs(ang))
   local cH = gW*math.sin(math.abs(ang)) + gH*math.cos(math.abs(ang))

   local sc_off =  2*math.max(cW/gW , cH/gH)
   --print('scaled by' ..sc_off)
   images = image.scale(images, '*'..sc_off)
   flow = flowX.scale(flow, sc_off, 'simple')

   -- rotate
   images = image.rotate(images, ang)
   flow = flowX.rotate(flow, ang)
   
   -- Now crop
   images = image.crop(images, "c", gW-2, gH-2)
   flow = image.crop(flow, "c", gW-2, gH-2)

   
   -- Add Random Scale
   local sc = torch.uniform(1.1, 1.7)

   images = image.scale(images, '*'..sc)
   flow = flowX.scale(flow, sc, 'simple')
   
   -- Add Random Noise to the images
   images = images:add(torch.rand(images:size()):mul(0.1):float())
      -- do random crop
   local iW = images:size(3)
   local iH = images:size(2)

   local oW = inputSize[3]
   local oH = inputSize[2]
   local h1 = math.floor(torch.uniform(1e-2, iH-oH))
   local w1 = math.floor(torch.uniform(1e-2, iW-oW))
   
   images = image.crop(images, w1, h1, w1 + oW, h1 + oH)
   flow = image.crop(flow, w1, h1, w1 + oW, h1 + oH)
   
   assert(images:size(3) == oW)
   assert(images:size(2) == oH)
   assert(flow:size(3) == oW)
   assert(flow:size(2) == oH)
   
   images = Preprocess()(images)

   return images, flow
end

--if paths.filep(trainCache) then
--   print('Loading train metadata from cache')
--   trainLoader = torch.load(trainCache)
--   trainLoader.sampleHookTrain = trainHook
   --assert(trainLoader.paths[1] == paths.concat(opt.data, 'train'),
   --       'cached files dont have the same path as opt.data. Remove your cached files at: '
   --          .. trainCache .. ' and rerun the program')
--else
   print('Creating train metadata')
   trainLoader = dataLoader{
      --paths = {paths.concat(opt.data, 'train')},
      inputSize = inputSize,
      flowSize = flowSize,
      samples = train_samples,
      verbose = true
   }
  -- torch.save(trainCache, trainLoader)
   trainLoader.sampleHookTrain = trainHook
   --trainLoader.samplingIds = train_samples
--end
collectgarbage()

--[[ do some sanity checks on trainLoader
do
   local class = trainLoader.imageClass
   local nClasses = #trainLoader.classes
   assert(class:max() <= nClasses, "class logic has error")
   assert(class:min() >= 1, "class logic has error")

end
--]]
-- End of train loader section
--------------------------------------------------------------------------------
--[[
   Section 2: Create a test data loader (testLoader),
   which can iterate over the test set and returns an image's
--]]

-- function to load the image
-- CHANGE THIS
local testInputSize = opt.testInputSize
local testFlowSize = opt.testOutputSize



local testHook = function(self, id)
   collectgarbage()
   local pathTable = stringx.split(self.samples[id], " ")
   local path1 = paths.concat(opt.data, pathTable[1])
   local path2 = paths.concat(opt.data, pathTable[2])
   local pathF = paths.concat(opt.data, pathTable[3])
   
   local img1 = loadImage(path1)
   local img2 = loadImage(path2)
   local flow = flowX.loadFLO(pathF)
   
   local images = torch.cat(img1, img2, 1)  
   
   images = TF.ColorNormalize(meanstd)(images)

   images = image.crop(images, "c", testInputSize[3], testInputSize[2])
   flow = image.crop(flow, "c", testFlowSize[3], testFlowSize[2])

   return images, flow
end

--if paths.filep(testCache) then
--   print('Loading test metadata from cache')
--   testLoader = torch.load(testCache)
--   testLoader.sampleHookTest = testHook
  -- assert(testLoader.paths[1] == paths.concat(opt.data, 'val'),
  --        'cached files dont have the same path as opt.data. Remove your cached files at: '
  --           .. testCache .. ' and rerun the program')
--else
   print('Creating test metadata')
   testLoader = dataLoader{
      --paths = {paths.concat(opt.data, 'val')},
      inputSize = testInputSize,
      flowSize = testFlowSize,
      samples = validation_samples,
      verbose = true
      --forceClasses = trainLoader.classes -- force consistent class indices between trainLoader and testLoader
   }
  -- torch.save(testCache, testLoader)
   testLoader.sampleHookTest = testHook
--   testLoader.samplingIds = validation_samples
--end
collectgarbage()
-- End of test loader section
