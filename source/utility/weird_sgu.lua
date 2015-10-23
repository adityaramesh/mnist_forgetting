local class = require "class"
weird_sgu = class("weird_sgu")

--
-- Note: the `model` parameter here is unused, but kept anyway to preserve API
-- uniformity. Other optimization algorithms may need to use this parameter to
-- perform model-specific operations (e.g. disabling/enabling dropout).
--
function weird_sgu:__init(model, state, logger)
	self.model       = model
	self.params      = model:parameters()
	self.grad_params = model:grad_parameters()
	self.state       = state or {}

	self.state.name = "weird_sgu"
	self.state.iter = self.state.iter          or {0, 0}
	self.lr         = self.state.learning_rate or lantern.schedule.constant(1e-3)
	self.mom        = self.state.momentum      or lantern.schedule.constant(0.95)
	self.mom_type   = self.state.momentum_type or lantern.momentum.none

	self.state.step = {}
end

function weird_sgu:update(n, batch)
	local iter = self.state.iter[n]
	self.state.iter[n] = self.state.iter[n] + 1

	local cur_lr = self.lr(iter)
	assert(cur_lr > 0 and cur_lr <= 1)

	if self.mom_type == lantern.momentum.none then
		local state = self.model:evaluate(n, batch)
		self.params[n]:add(-cur_lr, self.grad_params[n])
		return state
	elseif self.mom_type == lantern.momentum.nag then
		-- For the first iteration, we just take the direction of
		-- steepest descent.
		if not self.state.step[n] then
			local state = self.model:evaluate(n, batch)
			self.state.step[n] = self.grad_params[n]:clone():mul(-cur_lr)
			self.params[n]:add(self.state.step[n])
			return state
		end

		local cur_mom = self.mom(iter)
		assert(cur_mom > 0 and cur_mom < 1)

		-- Evaluate the function at the trial point.
		self.state.step[n]:mul(cur_mom)
		self.params[n]:add(self.state.step[n])
		local state = self.model:evaluate(n, batch)

		-- Update the parameters. We don't multiply the gradient by
		-- `-cur_lr` in advance because the logging function requires
		-- the original value.
		self.state.step[n]:add(-cur_lr, self.grad_params[n])
		self.params[n]:add(-cur_lr, self.grad_params[n])
		return state
	else
		error("Unsupported momentum type.")
	end
end
