require 'torch'
local totem = require 'totem'
local autograd = require 'autograd'
local util = require 'autograd.util'
local gradcheck = require 'autograd.gradcheck' {randomizeInput = true}
local gradcheckConstant = require 'autograd.gradcheck' {randomizeInput = false}
local tester = totem.Tester()
local stringx = require 'pl.stringx'

include('EPECriterion.lua')

local eps = 1e-12

local function epe(input, target)
-- TODO: Assertion for 4D tensor and appropriate flow fields
--	assert( input:nElement() == target:nElement(),
--    "input and target size mismatch")

    
    --buffer = input
    local npixels = torch.nElement(input)/2    -- 2 channel flow fields

    local buffer = torch.pow(torch.add(input, -1, target), 2)
    local output = torch.sqrt(torch.sum(buffer,2))   -- second channel is flow
    output = torch.sum(output)

    output = output / npixels

    return output    
end

local epeCriterion = nn.EPECriterion()
local autoepeCriterion = autograd.nn.AutoCriterion('AutoEPE')(epe)

for i=1,10 do
	local input = torch.rand(4,2,32,32)
	local target = torch.rand(4,2,32,32)

	local loss = epeCriterion:forward(input, target)
	local autoloss = autoepeCriterion:forward(input, target)
	local grads = epeCriterion:backward(input, target)
	local autograds = autoepeCriterion:backward(input, target)

	assert(torch.abs(loss - autoloss) < 1e-6, "Test Failed, Check Loss Function" )
	assert((grads - autograds):abs():max() < 1e-6, "Test Failed, Check Gradient Function" )

	print("Test " ..i .." Passed!")

end


