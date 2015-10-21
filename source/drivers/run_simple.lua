require "lantern"
--require "source/models/simple_net"
require "source/models/split_net"
--require "source/utility/masking_strategy"
require "source/utility/masking_strategy_split"
require "source/utility/alternating_update_driver"

local function extra_options(cmd)
	cmd:option("-train_file_1",     "",    "First training file.")
	cmd:option("-train_file_2",     "",    "Second training file.")
	cmd:option("-masking_strategy", "",    "none | first")
	cmd:option("-depth",            2,     "Depth of network.")
	cmd:option("-with_bn",          false, "Whether to use batch normalization.")
end

local info = lantern.parse_options(extra_options)
local opt = lantern.options

assert(string.len(opt.train_file_1) >= 1)
assert(opt.depth >= 2)

local train_files, outputs
if string.len(opt.train_file_2) == 0 then
	train_files = {opt.train_file_1}

	local data = torch.load(opt.train_file_1)
	outputs = data.classes
else
	train_files = {opt.train_file_1, opt.train_file_2}

	local data_1 = torch.load(opt.train_file_1)
	local data_2 = torch.load(opt.train_file_2)
	outputs = data_1.classes
	assert(outputs == data_2.classes)
end

local test_file
if outputs == 5 then
	test_file = "data/mnist/partitioned_8x8/test_task_2_reindexed.t7"
elseif outputs == 10 then
	test_file = "data/mnist/partitioned_8x8/test_task_2.t7"
else
	error("Unexpected number of outputs: " .. outputs .. ".")
end

local masking_strategy
if opt.masking_strategy == "first" then
	masking_strategy = mask_on_first_task
else
	assert(opt.masking_strategy == "none")
end

local bp = lantern.batch_provider({
	train_files       = train_files,
	test_file         = test_file,
	target            = "gpu",
	batch_size        = 50,
	sampling_strategy = "alternating"
})

local width = bp.train_data[1].inputs:size(2)
assert(width == bp.train_data[1].inputs:size(3))

--local model = simple_net(torch.LongStorage{width, width}, outputs, opt.depth,
--	opt.with_bn, masking_strategy)

local model = split_net(torch.LongStorage{width, width}, outputs, 50,
	opt.depth, opt.with_bn, masking_strategy)

local optim
if opt.depth == 2 then
	optim = lantern.optimizers.sgu(model, {
		learning_rate = lantern.schedule.gentle_decay(1e-3, 1e-5),
		momentum      = lantern.schedule.constant(0.95),
		momentum_type = lantern.momentum.nag
	})
elseif opt.depth == 5 then
	optim = lantern.optimizers.adadelta_lm(model, {
		learning_rate = lantern.schedule.constant(1),
		momentum_type = lantern.momentum.none
	})
else
	error("Don't know which optimizer to use for depth = " .. depth .. ".")
end

lantern.run({
	model        = info.model or model,
	driver       = alternating_update_driver(bp),
	-- Use this when gradient summation is not desired. Also look at the
	-- `evaluate` function of the model class in case further changes are
	-- necessary.
	--driver       = lantern.driver(bp),
	perf_metrics = {"accuracy", "gradient_norm"},
	model_dir    = info.model_dir,
	optimizer    = info.optimizer,
	history      = info.history,
	optimizer    = optim,
	stop_crit    = lantern.criterion.max_epochs(200000000)
})
