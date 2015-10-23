require "torch"
require "cunn"
require "nngraph"

local class = require "class"
weird_net = class("weird_net")

function weird_net:__init(input_shape, outputs)
	local inputs = 1
	for i = 1, #input_shape do
		inputs = inputs * input_shape[i]
	end

	assert(inputs > 0)
	assert(outputs > 0)

	self.input_shape   = input_shape
	self.output_shape_ = torch.LongStorage{outputs}

	local tie = function(src, dst)
		dst.data.module.weight = src.data.module.weight
		dst.data.module.bias = src.data.module.bias
	end

	local tie_transpose = function(src, dst)
		dst.data.module.weight = src.data.module.weight:t()
		dst.data.module.bias = src.data.module.bias
	end

	local input_1 = nn.View(inputs)()
	local input_2 = nn.View(inputs)()

	--[[
	        O    <--- input
	       / \   <--- expand
	      O   O  <--- l1, r1
	       \ /   <--- compress
	        O    <--- l2, r2
	       / \   <--- expand
	      O   O  <--- l3, r3
	       \ /   <--- compress
	        O    <--- l4, r4
	       / \   <--- reshape for softmax
	      O   O  <--- l5, r5
	      |   |
	      x   x  <--- softmax
	]]--

	-- The architecture below is the best one found so far. It gives 80.35%
	-- test accuracy.

	local l1 = nn.Linear(inputs, 512)(input_1)
	local u1 = nn.ReLU()(l1)
	local l2 = nn.Linear(512, 900)(u1)
	local u2 = nn.ReLU()(l2)

	local r1 = nn.Linear(inputs, 512)(input_2)
	local v1 = nn.ReLU()(r1)
	local r2 = nn.Linear(512, 900)(v1)
	local v2 = nn.ReLU()(r2)

	local l3 = nn.Linear(900, 1100)(u2)
	local u3 = nn.ReLU()(l3)
	local l4 = nn.Linear(1100, 1100)(u3)
	local u4 = nn.ReLU()(l4)

	local r3 = nn.Linear(900, 1100)(v2)
	local v3 = nn.ReLU()(r3)
	local r4 = nn.Linear(1100, 1100)(v3)
	local v4 = nn.ReLU()(r4)

	local l5 = nn.Linear(1100, outputs)(u4)
	local r5 = nn.Linear(1100, outputs)(v4)

	tie(l1, r1)
	--tie(l2, r2)
	tie(l3, r3)
	--tie(l4, r4)
	tie(l5, r5)

	local output_1 = nn.LogSoftMax()(l5)
	local output_2 = nn.LogSoftMax()(r5)

	self.model = {
		nn.gModule({input_1}, {output_1}):cuda(),
		nn.gModule({input_2}, {output_2}):cuda()
	}
	self.criterion = nn.ClassNLLCriterion():cuda()

	local p1, g1 = self.model[1]:getParameters()
	local p2, g2 = self.model[2]:getParameters()
	self.params = {p1, p2}
	self.grad_params = {g1, g2}
end

function weird_net:parameters()
	return self.params
end

function weird_net:grad_parameters()
	return self.grad_params
end

function weird_net:input_shape()
	return self.input_shape
end

function weird_net:output_shape()
	return self.output_shape_
end

function weird_net:predict(n, batch)
	local outputs = self.model[n]:forward(batch.inputs)
	local loss = self.criterion:forward(outputs, batch.targets)
	return {outputs = outputs, loss = loss}
end

function weird_net:evaluate(n, batch)
	self.grad_params[n]:zero()
	state = self:predict(n, batch)
	self.model[n]:backward(batch.inputs, self.criterion:backward(
		state.outputs, batch.targets))
	return state
end
