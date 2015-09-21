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

		-- Create the mini-batch.
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

		context.logger:log_value("train_epoch", info.train.epoch)
		context.logger:log_value("train_iter", (i - 1) / batch_size + 1)
		context.logger:flush()
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
	local valid_size = data.inputs:size(1)
	local batch_size = info.train.batch_size
	local params     = context.params
	local confusion  = context.confusion
	model:evaluate()

	print("Performing validation epoch.")
	for i = 1, valid_size, batch_size do
		xlua.progress(i, valid_size)

		-- Create the mini-batch.
		local cur_batch_size = math.min(batch_size, valid_size - i + 1)
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

	xlua.progress(valid_size, valid_size)
	confusion:updateValids()
	local acc = confusion.totalValid
	print("Mean class accuracy (validation): " .. 100 * acc .. "%.")
	confusion:zero()

	if info.train.epoch ~= nil then
		model_io.save_test_progress(function(x, y) return x > y end,
			info.train.epoch - 1, acc, paths, info)
	end
end

function make_context(info, class_count)
	local context = {}
	local log_dir = paths.concat("models", info.options.model)
	local log_path = paths.concat(log_dir, info.options.model .. "_output.log")
	context.logger = CSVLogger.create(log_path, {"train_epoch", "train_iter"})
		
	local model = info.model.model
	local criterion = info.model.criterion
	context.params, context.grad_params = model:getParameters()
	context.confusion = optim.ConfusionMatrix(class_count)

	context.grad_func = function(input, target, update_confusion)
		context.grad_params:zero()
		local output = model:forward(input)
		local loss = criterion:forward(output, target)

		if update_confusion then
			context.confusion:batchAdd(output, target)
		end

		model:backward(input, criterion:backward(output, target))
		return loss
	end

	context.optimizer = info.train.opt_method.create(
		model, context.params, context.grad_params, context.grad_func,
		info.train.opt_state)--, context.logger)
	context.logger:write_header()
	return context
end

function save_current_model(context, paths_, info)
	local epoch = info.train.epoch
	assert(epoch > 0)

	local output_fn = paths.concat(paths_.output_dir, "model_" .. (epoch - 1) .. "_epochs.t7")
	print("Saving current model to " .. output_fn .. ".")
	torch.save(output_fn, info.model.model)
end

function run(task_info_func, model_info_func, train_info_func)
	print("Loading data.")
	local train_data, valid_data, class_count = task_info_func()
	local do_train, _, paths, info = model_io.restore(
		model_info_func, train_info_func)

	-- Since the training epoch count is only incremented after
	-- serialization, the actual training epoch will always be one greater
	-- than the number that has been serialized.
	if info.train.epoch ~= nil then
		info.train.epoch = info.train.epoch + 1
	end

	local context = make_context(info, class_count)
	local valid_epoch_ratio = info.train.valid_epoch_ratio or 1
	print("")

	info.train.epoch = 1

	if do_train then
		while info.train.epoch <= 80 do
			-- At this point, `info.train.epoch` is the one-based
			-- index of the epoch that is *about* to be performed.
			if info.train.epoch % 10 == 1 then
				save_current_model(context, paths, info)
			end

			do_train_epoch(train_data, context, paths, info)
			print("")

			local cur_epoch = info.train.epoch - 1
			if valid_data ~= nil and cur_epoch % valid_epoch_ratio == 0 then
				do_valid_epoch(valid_data, context, paths, info)
				print("")
			end
		end
	elseif valid_data ~= nil then
		do_valid_epoch(valid_data, context, paths, info)
	end
end
