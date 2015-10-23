require "sys"

local class = require "class"
weird_net_driver = class("weird_net_driver")

--
-- The only argument to the constructor is the `batch_provider` instance used to
-- construct mini-batches from the data.
--
function weird_net_driver:__init(bp)
	self.bp = bp

	if bp.train_data then
		self.train = true
	else
		self.train = false
	end

	if bp.test_data then
		self.test = true
	else
		self.test = false
	end
end

function weird_net_driver:train_epoch(model, optim, acc, logger)
	assert(self.train)
	local sampler = self.bp:make_train_sampler()

	for i = 1, self.bp.train_batches do
		local start = sys.clock()
		local batch = sampler:next()
		local state

		if i % 2 == 0 then
			state = optim:update(2, batch)
		else
			state = optim:update(1, batch)
		end

		acc:update(batch, state)
		logger:update("/progress", {
			processed_instances = i,
			total_instances = self.bp.train_batches,
			time = sys.clock() - start
		})
	end
	return acc:value()
end

function weird_net_driver:test_epoch(model, optim, acc, logger)
	assert(self.test)
	local sampler = self.bp:make_test_sampler()

	for i = 1, self.bp.test_batches do
		local start = sys.clock()
		local batch = sampler:next()
		local state = model:predict(2, batch)
		acc:update(batch, state)

		-- Remember to update this
		logger:update("/progress", {
			processed_instances = i,
			total_instances = self.bp.test_batches,
			time = sys.clock() - start
		})
	end
	return acc:value()
end
