-- Copyright 2018 Anurag Ranjan and the Max Planck Gesellschaft.
-- All rights reserved.
-- This software is provided without any warranty.
-- By using this software you agree to the terms of the license file
-- in the root folder.
--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'optim'
--[[
   1. Setup SGD optimization state and learning rate schedule
   2. Create loggers.
   3. train - this function handles the high-level training loop,
              i.e. load data, train model, save model and state to disk
   4. trainBatch - Used by train() to train a single batch after the data is loaded.
]]--

-- Setup a reused optimization state (for sgd). If needed, reload it from disk
local optimState = {
    learningRate = opt.LR,
    learningRateDecay = opt.learningRateDecay,
    momentum = opt.momentum,
    dampening = 0.0,
    weightDecay = opt.weightDecay
}

if opt.optimState ~= 'none' then
    assert(paths.filep(opt.optimState), 'File not found: ' .. opt.optimState)
    print('Loading optimState from file: ' .. opt.optimState)
    optimState = torch.load(opt.optimState)
end

-- Learning rate annealing schedule. We will build a new optimizer for
-- each epoch.
--
-- By default we follow a known recipe for a 55-epoch training. If
-- the learningRate command-line parameter has been specified, though,
-- we trust the user is doing something manual, and will use her
-- exact settings for all optimization.
--
-- Return values:
--    diff to apply to optimState,
--    true IFF this is the first epoch of a new regime
local function paramsForEpoch(epoch)
    if opt.LR ~= 0.0 then -- if manually specified
        return { }
    end
    local regimes = {
        -- start, end,       LR,   WD,
        { 1,    25,   1e-4,   2e-4 },
        { 26,   50,   5e-5,   2e-4 },
        { 51,   75,   2e-5,   2e-4 },
        { 76, 100,   1e-5,   2e-4 },
        { 101, 1e8,   5e-6,   2e-4 },
    }

    for _, row in ipairs(regimes) do
        if epoch >= row[1] and epoch <= row[2] then
            return { learningRate=row[3], weightDecay=row[4] }, epoch == row[1]
        end
    end
end

-- 2. Create loggers.
trainLogger = optim.Logger(paths.concat(opt.save, 'train.log'))
local batchNumber
local lossF1_epoch, lossF2_epoch, lossF3_epoch, lossF4_epoch

-- 3. train - this function handles the high-level training loop,
--            i.e. load data, train model, save model and state to disk
function train()
   print('==> doing epoch on training data:')
   print("==> online epoch # " .. epoch)

   local params, newRegime = paramsForEpoch(epoch)
   if newRegime then
      optimState = {
         learningRate = params.learningRate,
         learningRateDecay = opt.learningRateDecay,
         momentum = opt.momentum,
         dampening = 0.0,
         weightDecay = params.weightDecay
      }
   end
   batchNumber = 0
   cutorch.synchronize()

   -- set the dropouts to training mode
   model:training()

   local tm = torch.Timer()
   lossF1_epoch = 0
   lossF2_epoch = 0
   lossF3_epoch = 0
   lossF4_epoch = 0

   for i=1,opt.epochSize do
      -- queue jobs to data-workers
      donkeys:addjob(
         -- the job callback (runs in data-worker thread)
         function()
            local inputs, flows = trainLoader:sample(opt.batchSize)
            return inputs, flows
         end,
         -- the end callback (runs in the main thread)
         trainBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   lossF1_epoch = lossF1_epoch / opt.epochSize
   lossF2_epoch = lossF2_epoch / opt.epochSize
   lossF3_epoch = lossF3_epoch / opt.epochSize
   lossF4_epoch = lossF4_epoch / opt.epochSize

   trainLogger:add{
      ['flow loss (train set)'] = lossF1_epoch,
      ['flow loss (train set)'] = lossF2_epoch,
      ['flow loss (train set)'] = lossF3_epoch,
      ['flow loss (train set)'] = lossF4_epoch
   }
   print(string.format('Epoch: [%d][TRAINING SUMMARY] Total Time(s): %.2f\t'
                          .. 'average flow loss (per batch): %.2f \t %.2f \t %.2f \t %.2f \t '
                          .. 'accuracy(%%):\t',
                       epoch, tm:time().real, lossF1_epoch, lossF2_epoch, lossF3_epoch, lossF4_epoch))
   print('\n')

   -- save model
   collectgarbage()

   -- clear the intermediate states in the model before saving to disk
   -- this saves lots of disk space
   model:clearState()
   saveDataParallel(paths.concat(opt.save, 'model_' .. epoch .. '.t7'), model) -- defined in util.lua
   torch.save(paths.concat(opt.save, 'optimState_' .. epoch .. '.t7'), optimState)
end -- of train()
-------------------------------------------------------------------------------------------
-- GPU inputs (preallocate)
local inputs = torch.CudaTensor()

local flows4 = torch.CudaTensor()
local flows3 = torch.CudaTensor()
local flows2 = torch.CudaTensor()
local flows1 = torch.CudaTensor()


local timer = torch.Timer()
local dataTimer = torch.Timer()

local parameters, gradParameters = model:getParameters()

-- 4. trainBatch - Used by train() to train a single batch after the data is loaded.
function trainBatch(inputsCPU, flowsCPU)
   cutorch.synchronize()
   collectgarbage()
   local dataLoadingTime = dataTimer:time().real
   timer:reset()

   -- scale flows

   -- transfer over to GPU
   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   flows1:resize(flowsCPU[1]:size()):copy(flowsCPU[1])
   flows2:resize(flowsCPU[2]:size()):copy(flowsCPU[2])
   flows3:resize(flowsCPU[3]:size()):copy(flowsCPU[3])
   flows4:resize(flowsCPU[4]:size()):copy(flowsCPU[4])


   local errF1, errF2, errF3, errF4, outputs

   feval = function(x)
      model:zeroGradParameters()

      outputs = model:forward(inputs) -- outputs = {flow, seg}
      errF1 = criterion1:forward(outputs[1], flows1)
      errF2 = criterion2:forward(outputs[2], flows2)
      errF3 = criterion3:forward(outputs[3], flows3)
      errF4 = criterion4:forward(outputs[4], flows4)

      local flowGradOutputs1 = criterion1:backward(outputs[1], flows1)
      local flowGradOutputs2 = criterion2:backward(outputs[2], flows2)
      local flowGradOutputs3 = criterion3:backward(outputs[3], flows3)
      local flowGradOutputs4 = criterion4:backward(outputs[4], flows4)

      model:backward(inputs, {flowGradOutputs1, flowGradOutputs2, flowGradOutputs3, flowGradOutputs4})
      err = {errF1, errF2, errF3, errF4}
      return err, gradParameters
   end
   optim.adam(feval, parameters, optimState)

   -- DataParallelTable's syncParameters
   if model.needsSync then
      model:syncParameters()
   end


   cutorch.synchronize()
   batchNumber = batchNumber + 1
   lossF1_epoch = lossF1_epoch + errF1
   lossF2_epoch = lossF2_epoch + errF2
   lossF3_epoch = lossF3_epoch + errF3
   lossF4_epoch = lossF4_epoch + errF4

   print(('Epoch: [%d][%d/%d]\tTime %.3f Flow Err %.4f  %.4f  %.4f %.4f LR %.0e DataLoadingTime %.3f'):format(
          epoch, batchNumber, opt.epochSize, timer:time().real, errF1,errF2,errF3,errF4,
          optimState.learningRate, dataLoadingTime))

   dataTimer:reset()
end
