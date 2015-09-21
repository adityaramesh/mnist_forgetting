package.path = package.path .. ";./torch_utils/?.lua"

require "source/task/svhn_small.lua"
require "source/models/cnn_3x3_seeded_v1.lua"
require "source/utility/run_model_v2.lua"
require "torch_utils/sopt"

function options_func(cmd)
	model_io.default_options(cmd)
	cmd:option("-max_epochs", 80, "Number of epochs for which to train" ..
		"current model.")
	cmd:option("-left_half_dir", "", "Directory for model whose left half " ..
		"was allowed to change.")
	cmd:option("-right_half_dir", "", "Directory for model whose right half " ..
		"was allowed to change.")
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

run(get_task_info, get_model_info, get_train_info, options_func)
