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
testLogger = optim.Logger(paths.concat(opt.save, 'test.log'))

local batchNumber
local lossF1, lossF2, lossF3, lossF4, lossF5
local timer = torch.Timer()

function test()
   print('==> doing epoch on validation data:')
   print("==> online epoch # " .. epoch)

   batchNumber = 0
   cutorch.synchronize()
   timer:reset()

   -- set the dropouts to evaluate mode
   model:evaluate()

   lossF1 = 0
   lossF2 = 0
   lossF3 = 0
   lossF4 = 0
   for i=1,nTest/opt.batchSize do -- nTest is set in 1_data.lua
      local indexStart = (i-1) * opt.batchSize + 1
      local indexEnd = (indexStart + opt.batchSize - 1)
      donkeys:addjob(
         -- work to be done by donkey thread
         function()
            local inputs, flows = testLoader:get(indexStart, indexEnd)
            return inputs, flows
         end,
         -- callback that is run in the main thread once the work is done
         testBatch
      )
   end

   donkeys:synchronize()
   cutorch.synchronize()

   lossF1 = lossF1 / (nTest/opt.batchSize) -- because loss is calculated per batch
   lossF2 = lossF2 / (nTest/opt.batchSize) -- because loss is calculated per batch
   lossF3 = lossF3 / (nTest/opt.batchSize) -- because loss is calculated per batch
   lossF4 = lossF4 / (nTest/opt.batchSize) -- because loss is calculated per batch
   testLogger:add{
      ['avg flow1 loss (test set)'] = lossF1,
      ['avg flow2 loss (test set)'] = lossF2,
      ['avg flow3 loss (test set)'] = lossF3,
      ['avg flow4 loss (test set)'] = lossF4
   }
   print(string.format('Epoch: [%d][TESTING SUMMARY] Total Time(s): %.2f \t'
                          .. 'average flow loss (per batch): %.2f \t %.2f \t %.2f \t %.2f \t ',
                       epoch, timer:time().real, lossF1, lossF2, lossF3, lossF4))

   print('\n')


end -- of test()
-----------------------------------------------------------------------------
local inputs = torch.CudaTensor()
local flows1 = torch.CudaTensor()
local flows2 = torch.CudaTensor()
local flows3 = torch.CudaTensor()
local flows4 = torch.CudaTensor()

function testBatch(inputsCPU, flowsCPU)
   batchNumber = batchNumber + opt.batchSize

   inputs:resize(inputsCPU:size()):copy(inputsCPU)
   flows1:resize(flowsCPU[1]:size()):copy(flowsCPU[1])
   flows2:resize(flowsCPU[2]:size()):copy(flowsCPU[2])
   flows3:resize(flowsCPU[3]:size()):copy(flowsCPU[3])
   flows4:resize(flowsCPU[4]:size()):copy(flowsCPU[4])

   local outputs = model:forward(inputs)
   --print('output 1 size', outputs[1]:size())
   --print('flow 1 size', flows1:size())
   local errF1 = criterion1:forward(outputs[1], flows1)
   local errF2 = criterion2:forward(outputs[2], flows2)
   local errF3 = criterion3:forward(outputs[3], flows3)
   local errF4 = criterion4:forward(outputs[4], flows4)

   cutorch.synchronize()
   --local pred = outputs:float()

   lossF1 = lossF1 + errF1
   lossF2 = lossF2 + errF2
   lossF3 = lossF3 + errF3
   lossF4 = lossF4 + errF4

   --local _, pred_sorted = pred:sort(2, true)
   --for i=1,pred:size(1) do
   --   local g = labelsCPU[i]
   --   if pred_sorted[i][1] == g then error_center = error_center + 1 end
   -- end

   --error_center = torch.pow(outputs - labelsCPU, 2):sum()


   --if batchNumber % 1024 == 0 then
   print(('Epoch: Testing [%d][%d/%d]'):format(epoch, batchNumber, nTest))
   --end
end
