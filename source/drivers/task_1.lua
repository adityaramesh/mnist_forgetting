require "lantern"
require "source/models/simple_net"
require "source/utility/masking_strategy"

local function extra_options(cmd)
	cmd:option("-train_file_1",     "", "First training file.")
	cmd:option("-train_file_2",     "", "Second training file.")
	cmd:option("-masking_strategy", "", "none | first | both")
	cmd:option("-depth",            2,  "Depth of network.")
	cmd:option("-outputs",          10, "Number of outputs of network.")
end

local info = lantern.parse_options(extra_options)
local opt = lantern.options

assert(string.len(opt.train_file_1) >= 1)
assert(opt.depth >= 2)
assert(opt.outputs == 5 or opt.outputs == 10)

local train_files
if string.len(opt.train_file_2) == 0 then
	train_files = {opt.train_file_1}
else
	train_files = {opt.train_file_1, opt.train_file_2}
end

local masking_strategy
if opt.masking_strategy == "first" then
	masking_strategy = mask_on_first_task()
elseif opt.masking_strategy == "both" then
	masking_strategy = mask_on_both_tasks()
else
	assert(opt.masking_strategy == "none")
end

local bp = lantern.batch_provider({
	train_files       = train_files,
	test_file         = "data/mnist/partitioned_8x8/test_task_2.t7",
	target            = "gpu",
	batch_size        = 200,
	sampling_strategy = "alternating"
})

local model = simple_net(torch.LongStorage{8, 8}, opt.outputs, opt.depth,
	masking_strategy)

local optim
if opt.depth == 2 then
	optim = lantern.optimizers.sgu(model, {
		-- XXX change this based on the masking strategy
		learning_rate = lantern.schedule.gentle_decay(1e-2, 1e-4),
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
	driver       = lantern.driver(bp),
	perf_metrics = {"accuracy"},
	model_dir    = info.model_dir,
	optimizer    = info.optimizer,
	history      = info.history,
	optimizer    = optim,
	stop_crit    = lantern.criterion.max_epochs(50)
})
