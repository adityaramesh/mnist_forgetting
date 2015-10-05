require "torch"
require "optim"
require "xlua"
require "torch_utils/utility"

local function do_train_epoch(data, context, paths, info)
	local model       = info.model.model
	local criterion   = info.model.criterion
	local params      = context.params
	local grad_params = context.grad_params
	local confusion   = context.confusion

	local train_size  = data.inputs:size(1)
	local opt_method  = info.train.opt_method
	local opt_state   = info.train.opt_state
	local batch_size  = info.train.batch_size

	local perm = torch.randperm(train_size)
	model:training()
	print("Starting training epoch " .. info.train.epoch .. ".")

	for i = 1, train_size, batch_size do
		xlua.progress(i, train_size)

		-- Create the mini-batch. Note: I measured the time needed to
		-- do this earlier, and it is insignificant.
		local cur_batch_size = math.min(batch_size, train_size - i + 1)
		local inputs = {}
		local targets = {}
		for j = i, i + cur_batch_size - 1 do
			local input = data.inputs[{{perm[j]}}]
			local target = data.targets[{{perm[j]}}]
			table.insert(inputs, input)
			table.insert(targets, target)
		end

		input = nn.JoinTable(1):forward(inputs):typeAs(params)
		target = nn.JoinTable(1):forward(targets):typeAs(params)
		context.optimizer:update(input, target)
	end

	xlua.progress(train_size, train_size)
	confusion:updateValids()
	local acc = confusion.totalValid
	print("Mean class accuracy (training): " .. 100 * acc .. "%.")
	confusion:zero()

	model_io.save_train_progress(function(x, y) return x > y end,
		info.train.epoch, acc, paths, info)
	info.train.epoch = info.train.epoch + 1
end

local function do_valid_epoch(data, context, paths, info)
	local model      = info.model.model
	local criterion  = info.model.criterion
	local test_size  = data.inputs:size(1)
	local batch_size = info.train.batch_size
	local params     = context.params
	local confusion  = context.confusion
	model:evaluate()

	print("Performing validation epoch.")
	for i = 1, test_size, batch_size do
		xlua.progress(i, test_size)

		-- Create the mini-batch. Note: I measured the time needed to
		-- do this earlier, and it is insignificant.
		local cur_batch_size = math.min(batch_size, test_size - i + 1)
		local inputs = {}
		local targets = {}
		for j = i, i + cur_batch_size - 1 do
			local input = data.inputs[{{j}}]
			local target = data.targets[{{j}}]
			table.insert(inputs, input)
			table.insert(targets, target)
		end

		inputs = nn.JoinTable(1):forward(inputs):typeAs(params)
		targets = nn.JoinTable(1):forward(targets):typeAs(params)
		local outputs = model:forward(inputs)
		confusion:batchAdd(outputs, targets)
	end

	xlua.progress(test_size, test_size)
	confusion:updateValids()
	local acc = confusion.totalValid
	print("Mean class accuracy (validation): " .. 100 * acc .. "%.")
	confusion:zero()

	if info.train.epoch ~= nil then
		model_io.save_test_progress(function(x, y) return x > y end,
			info.train.epoch - 1, acc, paths, info)
	end
end

local function zero_half_grad_params(opt, layer)
	local _, grad_params = layer:parameters()

	for i = 1, #grad_params do
		local group = grad_params[i]
		-- We expect the number of feature maps in each of the
		-- convolutional layers to be even.
		assert(group:size(1) % 2 == 0)

		if opt.half == 1 then
			group[{{1, group:size(1) / 2}}]:zero()
		elseif opt.half == 2 then
			group[{{group:size(1) / 2 + 1, group:size(1)}}]:zero()
		else
			error("Invalid value for `opt.half`: " .. opt.half .. ".")
		end
	end
end

function mask_gradient(opt, model)
	-- XXX: this is special cased to `cnn_5x5_mnist`.
	--local layers = {1, 2, 5, 6, 9, 10}
	-- This is for the 8x8 net.
	local layers = {2, 4}

	for i = 1, #layers do
		zero_half_grad_params(opt, model:get(layers[i]))
	end
end

function make_context(info, class_count)
	local context = {}
	local model = info.model.model
	local criterion = info.model.criterion
	context.params, context.grad_params = model:getParameters()
	context.confusion = optim.ConfusionMatrix(class_count)

	if info.train.epoch == nil then
		info.train.epoch = 1
	else
		-- Since the training epoch count is only incremented after
		-- serialization, the actual training epoch will always be
		-- one greater than the number that has been serialized.
		info.train.epoch = info.train.epoch + 1
	end

	context.grad_func = function(input, target, update_confusion)
		context.grad_params:zero()
		local output = model:forward(input)
		local loss = criterion:forward(output, target)

		if update_confusion then
			context.confusion:batchAdd(output, target)
		end

		model:backward(input, criterion:backward(output, target))
		mask_gradient(info.options, model)
		return loss
	end

	context.optimizer = info.train.opt_method.create(
		model, context.params, context.grad_params, context.grad_func,
		info.train.opt_state)
	return context
end

function run_model(task_info_func, model_info_func, train_info_func, options_func)
	local do_train, _, paths, info = model_io.restore(
		model_info_func, train_info_func, options_func)

	print("Loading data.")
	local train_data, test_data = task_info_func(info.options)
	assert(train_data.classes == test_data.classes)

	local context = make_context(info, train_data.classes)
	local max_epochs = info.train.max_epochs
	local valid_epoch_ratio = info.train.valid_epoch_ratio or 1
	print("")

	if do_train then
		while not max_epochs or (max_epochs and info.train.epoch <= max_epochs) do
			do_train_epoch(train_data, context, paths, info)
			print("")

			local cur_epoch = info.train.epoch - 1
			if cur_epoch % valid_epoch_ratio == 0 then
				do_valid_epoch(test_data, context, paths, info)
				print("")
			end
		end
	else
		do_valid_epoch(test_data, context, paths, info)
	end
end
