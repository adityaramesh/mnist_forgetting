require "torch"
require "cunn"
require "nngraph"

local class = require "class"
split_net = class("split_net")

function split_net:__init(input_shape, outputs, batch_size, depth, use_bn, grad_mod_func)
	local inputs = 1
	for i = 1, #input_shape do
		inputs = inputs * input_shape[i]
	end

	assert(inputs > 0)
	assert(outputs > 0)
	assert(depth >= 1)

	self.input_shape   = input_shape
	self.output_shape_ = torch.LongStorage{outputs}
	self.grad_mod_func = grad_mod_func

	local input = nn.View(inputs)()
	local output_1, output_2

	if depth == 1 then
		output_1 = nn.LogSoftMax()(nn.Linear(inputs, outputs)(input))
		output_2 = nn.LogSoftMax()(nn.Linear(inputs, outputs)(input))
	else
		local a = nn.Linear(inputs, 512)(input)
		local b = a

		if use_bn then b = nn.BatchNormalization(512)(a) end
		local c = nn.ReLU()(b)

		for i = 1, depth - 2 do
			a = nn.Linear(512, 512)(c)
			b = a

			if use_bn then b = nn.BatchNormalization(512)(a) end
			c = nn.ReLU()(b)
		end

		output_1 = nn.LogSoftMax()(nn.Linear(512, outputs)(c))
		output_2 = nn.LogSoftMax()(nn.Linear(512, outputs)(c))
	end

	self.model = nn.gModule({input}, {output_1, output_2}):cuda()
	self.criterion = nn.ClassNLLCriterion():cuda()
	self.params, self.grad_params = self.model:getParameters()

	-- Storage for temporary data.
	self.prev_grads = torch.CudaTensor(batch_size, outputs)
end

function split_net:parameters()
	return self.params
end

function split_net:grad_parameters()
	return self.grad_params
end

function split_net:input_shape()
	return self.input_shape
end

function split_net:output_shape()
	return self.output_shape_
end

function split_net:predict(batch)
	local outputs = self.model:forward(batch.inputs)[2]
	local loss = self.criterion:forward(outputs, batch.targets)
	return {outputs = outputs, loss = loss}
end

function split_net:evaluate(batch)
	if batch.dataset == 1 then
		self.grad_params:zero()
	end

	local state
	if batch.dataset == 1 then
		local outputs = self.model:forward(batch.inputs)[1]
		local loss = self.criterion:forward(outputs, batch.targets)

		local grads = self.criterion:backward(outputs, batch.targets)
		self.prev_grads:zero()
		self.prev_grads[{{1, grads:size(1)}}]:copy(grads)
		state = {outputs = outputs, loss = loss}
	else
		local outputs = self.model:forward(batch.inputs)[2]
		local loss = self.criterion:forward(outputs, batch.targets)
		state = {outputs = outputs, loss = loss}

		self.model:backward(batch.inputs, {
			self.prev_grads,
			self.criterion:backward(outputs, batch.targets)
		})
	end

	return state

	-- Use this when gradient summation is not desired.
	--self.grad_params:zero()

	--local state
	--if batch.dataset == 1 then
	--	local outputs = self.model:forward(batch.inputs)[1]
	--	local loss = self.criterion:forward(outputs, batch.targets)
	--	state = {outputs = outputs, loss = loss}

	--	self.model:backward(batch.inputs, {
	--		self.criterion:backward(outputs, batch.targets),
	--		self.prev_grads
	--	})
	--else
	--	local outputs = self.model:forward(batch.inputs)[2]
	--	local loss = self.criterion:forward(outputs, batch.targets)
	--	state = {outputs = outputs, loss = loss}

	--	self.model:backward(batch.inputs, {
	--		self.prev_grads,
	--		self.criterion:backward(outputs, batch.targets)
	--	})
	--	self.grad_params:mul(5)
	--end

	--return state
end
