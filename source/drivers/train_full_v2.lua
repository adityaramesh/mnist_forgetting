package.path = package.path .. ";./torch_utils/?.lua"

require "source/task/svhn_small.lua"
require "source/models/cnn_3x3.lua"
require "source/utility/run_model_v2.lua"
require "torch_utils/sopt"

function options_func(cmd)
	model_io.default_options(cmd)
	cmd:option("-max_epochs", 80, "Number of epochs for which to train" ..
		"current model.")
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
