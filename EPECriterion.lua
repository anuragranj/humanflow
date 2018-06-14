local EPECriterion, parent = torch.class('nn.EPECriterion', 'nn.Criterion')

-- Computes average endpoint error for batchSize x ChannelSize x Height x Width
-- flow fields or general multidimensional matrices.

local  eps = 1e-12

function EPECriterion:__init()
	parent.__init(self)
	self.sizeAverage = true
end

function EPECriterion:updateOutput(input, target)
-- TODO: Assertion for 4D tensor and appropriate flow fields
	assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local output
    local npixels

    buffer:resizeAs(input)
    npixels = input:nElement()/2    -- 2 channel flow fields

    buffer:add(input, -1, target):pow(2)
    output = torch.sum(buffer,2):sqrt()   -- second channel is flow
    output = output:sum()

    output = output / npixels

    self.output = output

    return self.output    
end

function EPECriterion:updateGradInput(input, target)

	assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local gradInput = self.gradInput
    local npixels
    local loss

    buffer:resizeAs(input)
    npixels = input:nElement()/2

    buffer:add(input, -1, target):pow(2)
    loss = torch.sum(buffer,2):sqrt():add(eps)  -- forms the denominator
    loss = torch.cat(loss, loss, 2)   -- Repeat tensor to scale the gradients

    gradInput:resizeAs(input)
    gradInput:add(input, -1, target):cdiv(loss)
-- TODO: scale it appropriately to account for Average Error
    gradInput = gradInput / npixels  
    return gradInput
end