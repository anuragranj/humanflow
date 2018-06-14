--
--  Copyright (c) 2014, Facebook, Inc.
--  All rights reserved.
--
--  This source code is licensed under the BSD-style license found in the
--  LICENSE file in the root directory of this source tree. An additional grant
--  of patent rights can be found in the PATENTS file in the same directory.
--
local M = { }

function M.parse(arg)
    local cmd = torch.CmdLine()
    cmd:text()
    cmd:text('Torch-7 Unsupervised Optical Flow Training')
    cmd:text()
    cmd:text('Options:')
    ------------ General options --------------------

    cmd:option('-cache', 'checkpoint/', 'subdirectory in which to save/log experiments')
    cmd:option('-data', '../AllFlowData/body_flows_v2', 'Training data directory')
    cmd:option('-trainFile', 'train.txt', 'Virtual Kitti Training Set')
    cmd:option('-valFile', 'val.txt', 'Virtual Kitti Validation Set')
    cmd:option('-manualSeed',        2, 'Manually set RNG seed')
    cmd:option('-GPU',                1, 'Default preferred GPU')
    cmd:option('-nGPU',               2, 'Number of GPUs to use by default')
    cmd:option('-backend',     'cudnn', 'Options: cudnn | ccn2 | cunn')
    ------------- Data options ------------------------
    cmd:option('-nDonkeys',        4, 'number of donkeys to initialize (data loading threads)')
    ------------- Training options --------------------
    cmd:option('-augment',         1,     'augment the data')   
    cmd:option('-nEpochs',         1000,    'Number of total epochs to run')
    cmd:option('-epochSize',       1000, 'Number of batches per epoch')
    cmd:option('-epochNumber',     1,     'Manual epoch number (useful on restarts)')
    cmd:option('-batchSize',       8,   'mini-batch size (1 = pure stochastic)')
    ---------- Optimization options ----------------------
    cmd:option('-LR',    0.0, 'learning rate; if set, overrides default LR/WD recipe')
    cmd:option('-momentum',        0.9,  'momentum')
    cmd:option('-weightDecay',     2e-4, 'weight decay')
    cmd:option('-learningRateDecay',     1e-7, 'learning rate decay')
    ---------- Model options ----------------------------------
    cmd:option('-netType',   'fullBodyModel', 'Options: volcon | lapgan | flownet | flownet2 ')
    cmd:option('-retrain',     'none', 'provide path to model to retrain with')
    cmd:option('-optimState',  'none', 'provide path to an optimState to reload from')
    cmd:text()

    --TODO : Remove preprocessing and augmentation options

    local opt = cmd:parse(arg or {})
    -- add commandline specified options
    opt.save = paths.concat(opt.cache)--,
    --                        cmd:string(opt.netType, opt,
    --                                   {netType=true, retrain=true, optimState=true, cache=true, data=true}))
    -- add date/time
    opt.save = paths.concat(opt.save, '' .. os.date():gsub(' ',''))
    -- This the input output size of the database

    opt.inputSize = {6, 256, 256}
    opt.outputSize = {2, 256, 256}

    opt.testInputSize = {6, 256, 256}
    opt.testOutputSize = {2, 256, 256}

    return opt
end

return M
