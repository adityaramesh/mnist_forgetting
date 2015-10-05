package.path = package.path .. ";./torch_utils/?.lua"
package.path = package.path .. ";./image_utils/?.lua"

require "cutorch"
require "torch_utils/sopt"
require "image_utils/image_utils"

function options_func(cmd)
	model_io.default_options(cmd)
	cmd:option("-train_file", "", "Path to file with training data.")
	cmd:option("-test_file", "", "Path to file with testing data.")
	cmd:option("-max_epochs", 80, "Number of epochs to train.")
	cmd:option("-valid_epoch_ratio", 1, "Number of training epochs per validation epoch.")
	cmd:option("-model_file", "source/models/cnn_5x5_mnist_fused.lua", "Path to file that defines model architecture.")
	cmd:option("-left_half_dir", "", "Directory with left half trained.")
	cmd:option("-right_half_dir", "", "Directory with right half trained.")
end

function get_task_info(opt)
	-- We need to add a dummy "channel" dimension so that
	-- `nn.SpatialConvolutionMM` works.
	local test_data = image_utils.load(opt.test_file)
	test_data.inputs = test_data.inputs:reshape(
		test_data.inputs:size(1), 1,
		test_data.inputs:size(2),
		test_data.inputs:size(3))

	if opt.task ~= "evaluate" then
		local train_data = image_utils.load(opt.train_file)
		train_data.inputs = train_data.inputs:reshape(
			train_data.inputs:size(1), 1,
			train_data.inputs:size(2),
			train_data.inputs:size(3))
		return train_data, test_data
	end

	-- If the model is only being evaluated on the test set, then there is
	-- no need to load the training data.
	return nil, test_data
end

function load_model_info(opt)
	dofile(opt.model_file)
	return get_model_info(opt)
end

function get_train_info(opt)
	return {
		opt_state = {
			learning_rate = sopt.constant(1),
			epsilon = 1e-11,
			decay = sopt.constant(0.95),
			momentum_type = sopt.none
		},
		opt_method = AdaDeltaLMOptimizer,
		batch_size = 200,
		max_epochs = opt.max_epochs,
		valid_epoch_ratio = opt.valid_epoch_ratio
	}
end

image_utils.run_model(get_task_info, load_model_info, get_train_info, options_func)
