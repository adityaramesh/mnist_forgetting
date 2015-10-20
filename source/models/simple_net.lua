require "torch"
require "cunn"

local class = require "class"
simple_net = class("simple_net")

function simple_net:__init(input_shape, outputs, depth, grad_mod_func)
	local inputs = 1
	for i = 1, #input_shape do
		inputs = inputs * input_shape[i]
	end

	assert(inputs > 0)
	assert(outputs > 0)
	assert(depth >= 1)

	self.input_shape   = input_shape
	self.output_shape_ = torch.LongStorage{outputs}
	self.model         = nn.Sequential()
	self.grad_mod_func = grad_mod_func

	self.model:add(nn.View(inputs))

	if depth == 1 then
		self.model:add(nn.Linear(inputs, outputs))
	else
		self.model:add(nn.Linear(inputs, 512))
		self.model:add(nn.BatchNormalization(512))
		self.model:add(nn.ReLU())

		for i = 1, depth - 2 do
			self.model:add(nn.Linear(512, 512))
			self.model:add(nn.BatchNormalization(512))
			self.model:add(nn.ReLU())
		end

		self.model:add(nn.Linear(512, outputs))
	end

	self.model:add(nn.LogSoftMax())
	self.model:cuda()
	self.criterion = nn.ClassNLLCriterion():cuda()
	self.params, self.grad_params = self.model:getParameters()
end

function simple_net:parameters()
	return self.params
end

function simple_net:grad_parameters()
	return self.grad_params
end

function simple_net:input_shape()
	return self.input_shape
end

function simple_net:output_shape()
	return self.output_shape_
end

function simple_net:predict(batch)
	local outputs = self.model:forward(batch.inputs)
	local loss = self.criterion:forward(outputs, batch.targets)
	return {outputs = outputs, loss = loss}
end

function simple_net:evaluate(batch)
	local state = self:predict(batch)

	-- XXX
	if batch.targets[1] <= 5 then
		self.grad_params:zero()
	end

	-- XXX
	--self.grad_params:zero()
	self.model:backward(batch.inputs, self.criterion:backward(
		state.outputs, batch.targets))

	--print(batch.targets[1] <= 5, self.grad_params:norm())
	--print(batch.targets[1] <= 5, self.grad_params:mean(), self.grad_params:std(), self.grad_params:norm())

	--local gind = torch.ge(self.grad_params, self.grad_params:mean() + 30 * self.grad_params:std())
	--local lind = torch.le(self.grad_params, self.grad_params:mean() - 30 * self.grad_params:std())
	--local gcount = gind:sum()
	--local lcount = lind:sum()

	--print(batch.targets[1] <= 5, gcount, lcount, self.grad_params:norm())

	--if batch.targets[1] >= 6 and gcount > 0 then
	--	--print(self.grad_params[gind])
	--	print(torch.nonzero(gind:float()))
	--end

	--if batch.targets[1] >= 6 and lcount > 0 then
	--	--print(self.grad_params[lind])
	--	print(torch.nonzero(lind:float()))
	--end

	if self.grad_mod_func then
		self.grad_mod_func(self, batch)
	end
	return state
end
