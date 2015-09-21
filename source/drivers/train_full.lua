package.path = package.path .. ";./torch_utils/?.lua"

require "source/task/svhn_small.lua"
require "source/models/cnn_3x3.lua"
require "source/utility/run_model_incremental_save.lua"
require "torch_utils/sopt"

function get_train_info()
        return {
                opt_state = {
                        learning_rate = sopt.constant(1),
			epsilon = 1e-11,
                        decay = sopt.constant(0.95),
                        momentum_type = sopt.none,
                },
                opt_method = AdaDeltaLMOptimizer,
                batch_size = 200,
		valid_epoch_ratio = 1
        }
end

run(get_task_info, get_model_info, get_train_info)
