-- Script to precompute Optical Flow
require 'nn'
require 'cunn'
require 'cudnn'

cmd = torch.CmdLine()
cmd:text()
cmd:text('Precompute Optical Flow at a level')
cmd:option('-level', 0, 'Pyramid Level')
cmd:option('-size', 22872, 'Number of Data Points')
cmd:option('-data', '../FlowNet2/data' , 'Data Directory')
cmd:text()
opt = cmd:parse(arg or {})

local model, modelpath
if opt.level == 0 then
	modelpath = paths.concat('models', 'model_00.t7')
	opt.fineHeight = 12
	opt.fineWidth = 16
	opt.saveDir = 'data16'

end

model = torch.load(modelpath)
if torch.type(model) == 'nn.DataParallelTable' then
	model = model:get(1)
end
model:evaluate()

local donkey = require('minidonkey')
local loss = 0
for i=1,opt.size do
	local id = i
	local imgs, flow = donkey.testHook(id)

	imgs:resize(1,8, opt.fineHeight, opt.fineWidth)

	local  flow_est = model:forward(imgs:cuda())

	local _err = (flow:cuda() - flow_est):pow(2)
    local err = torch.sum(_err, 2):sqrt():float()
    
    loss = loss + (err:sum()/(opt.fineHeight*opt.fineWidth))
    print('Error = ' .. err:sum()/(opt.fineHeight*opt.fineWidth))
    print('Loss = ' .. loss/i)

	donkey.saveHook(id, flow_est:squeeze():float(), opt.saveDir) 
end