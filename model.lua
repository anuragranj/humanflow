--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
require 'nn'
require 'cunn'
require 'cudnn'
require 'optim'
require 'nngraph'
require 'stn'
include('EPECriterion.lua')
--[[
   1. Create Model
   2. Create Criterion
   3. Convert model to CUDA
]]--

-- 1. Create Network
-- 1.1 If preloading option is set, preload weights from existing models appropriately
if opt.retrain ~= 'none' then
   assert(paths.filep(opt.retrain), 'File not found: ' .. opt.retrain)
   print('Loading model from file: ' .. opt.retrain);
   model = loadDataParallel(opt.retrain, opt.nGPU) -- defined in util.lua
else
   model = paths.dofile('models/' .. opt.netType .. '.lua')
   print('=> Creating model from file: models/' .. opt.netType .. '.lua')
end

-- Load Histogram weights
criterion1 = nn.EPECriterion():cuda()
criterion2 = nn.EPECriterion():cuda()
criterion3 = nn.EPECriterion():cuda()
criterion4 = nn.EPECriterion():cuda()
criterion5 = nn.EPECriterion():cuda()

print('=> Model')
--print(model)

print('=> Criterion')
print(criterion1)

collectgarbage()
