require "sys"

local class = require "class"
alternating_update_driver = class("alternating_update_driver")

--
-- The only argument to the constructor is the `batch_provider` instance used to
-- construct mini-batches from the data.
--
function alternating_update_driver:__init(bp)
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

function alternating_update_driver:train_epoch(model, optim, acc, logger)
	assert(self.train)
	local sampler = self.bp:make_train_sampler()
	local count = self.bp.train_batches - self.bp.train_batches % 2

	for i = 1, count do
		if i % 2 == 0 then
			local start = sys.clock()
			local batch = sampler:next()
			local state = optim:update(batch)
			acc:update(batch, state)

			logger:update("/progress", {
				processed_instances = i / 2,
				total_instances = count / 2,
				time = sys.clock() - start
			})
		else
			local batch = sampler:next()
			model:evaluate(batch)
		end
	end
	return acc:value()
end

function alternating_update_driver:test_epoch(model, optim, acc, logger)
	assert(self.test)
	local sampler = self.bp:make_test_sampler()

	for i = 1, self.bp.test_batches do
		local start = sys.clock()
		local batch = sampler:next()
		local state = model:predict(batch)
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
