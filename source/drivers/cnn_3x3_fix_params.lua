package.path = package.path .. ";./torch_utils/?.lua"

require "torch"
require "nn"
require "cunn"

require "source/task/svhn_small.lua"
require "source/utility/run_model_fix_params.lua"
require "torch_utils/sopt"

function options_func(cmd)
	model_io.default_options(cmd)
	cmd:option("-pretrained_epochs", 0, "Number of epochs for which resumed" ..
		"model was trained.")
	cmd:option("-max_epochs", 20, "Number of epochs for which to train" ..
		"current model.")
	cmd:option("-half", 1, "The half of the weights in each convolutional" ..
		"layer that should be set to zero.")
end

function get_model_info(opt)
	local model_fn = "model_" .. opt.pretrained_epochs .. "_epochs.t7"
	local model_path = paths.concat("models/cnn_3x3_full", model_fn)

	print("Loading model " .. model_fn .. ".")
	local model = torch.load(model_path)

	return {
		model = model:cuda(),
		criterion = nn.ClassNLLCriterion():cuda()
	}
end

function get_train_info(opt)
        return {
                opt_state = {
                        learning_rate = sopt.constant(1),
			epsilon = 1e-11,
                        decay = sopt.constant(0.95),
                        momentum_type = sopt.none,
                },
                opt_method = AdaDeltaLMOptimizer,
                batch_size = 200,
		valid_epoch_ratio = 1,
		max_epochs = opt.max_epochs
        }
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

function modify_grad_func(opt, model)
	local layers = {2, 5, 9, 12, 16, 19}
	for i = 1, #layers do
		zero_half_grad_params(opt, model:get(layers[i]))
	end
end

run(get_task_info, get_model_info, get_train_info, options_func, modify_grad_func)
