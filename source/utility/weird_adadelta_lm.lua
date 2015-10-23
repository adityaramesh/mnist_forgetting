--
-- A version of AdaDelta that uses less memory, but has the following disadvantages:
-- * Does not support logging. Logging slows things down and uses up extra
--   memory anyway, so if you need support for this, then use
--   `lantern.optimizers.adadelta`.
-- * May suffer from numerical issues in certain situations. If you suspect that
--   this is happening to you, try `lantern.optimizers.adadelta`.
--

local class = require 'class'
weird_adadelta_lm = class("weird_adadelta_lm")

function weird_adadelta_lm:__init(model, state)
	self.model       = model
	self.params      = model:parameters()
	self.grad_params = model:grad_parameters()
	self.state       = state or {}

	self.state.name = "weird_adadelta_lm"
	self.state.iter = self.state.iter          or {0, 0}
	self.eps        = self.state.eps           or 1e-10
	self.lr         = self.state.learning_rate or lantern.schedule.constant(1e-3)
	self.mom        = self.state.momentum      or lantern.schedule.constant(0.95)
	self.decay      = self.state.decay         or lantern.schedule.constant(0.95)
	self.mom_type   = self.state.momentum_type or lantern.momentum.none

	self.state.temp = {}
	self.state.grad_mom_2 = {}
	self.state.update_mom_2 = {}
end

function weird_adadelta_lm:update(n, batch)
	local iter = self.state.iter[n]
	self.state.iter[n] = self.state.iter[n] + 1

	local cur_lr = self.lr(iter)
	local cur_decay = self.decay(iter)
	assert(cur_lr > 0 and cur_lr <= 1)
	assert(cur_decay > 0 and cur_decay < 1)

	-- Initializing the parameters here causes the first update to be
	-- multiplied by `(1 - cur_decay)`, since the running average of the
	-- second moment estimates will be zero. While it may seem like using a
	-- severe underestimate may impede convergence, I have actually found
	-- that the optimizer converges faster this way.
	if not self.state.temp[n] then
		-- Used as a buffer to store intermediate values.
		self.state.temp[n] = torch.Tensor():typeAs(self.params[n]):
			resizeAs(self.params[n]):zero()
		-- Estimate of the second moment of the gradient.
		self.state.grad_mom_2[n] = torch.Tensor():typeAs(self.params[n]):
			resizeAs(self.params[n]):zero()
		-- Estimate of the second moment of the update.
		self.state.update_mom_2[n] = torch.Tensor():typeAs(self.params[n]):
			resizeAs(self.params[n]):zero()
	end

	local state = self.model:evaluate(n, batch)
	self.state.temp[n]:pow(self.grad_params[n], 2)
	self.state.grad_mom_2[n]:mul(cur_decay):add(1 - cur_decay, self.state.temp[n])

	-- Note: adding and subtracting eps from the same quantity will
	-- not result in a contribution of zero in general. This may
	-- cause issues in certain situations.

	self.state.update_mom_2[n]:add(self.eps)
	self.state.grad_mom_2[n]:add(self.eps)
	self.state.temp[n]:cdiv(self.state.update_mom_2[n], self.state.grad_mom_2[n]):
		sqrt():cmul(self.grad_params[n]):mul(-cur_lr)
	self.state.update_mom_2[n]:add(-self.eps)
	self.state.grad_mom_2[n]:add(-self.eps)
	self.params[n]:add(self.state.temp[n])

	self.state.temp[n]:pow(2):mul(1 - cur_decay)
	self.state.update_mom_2[n]:mul(cur_decay):add(self.state.temp[n])

	return state
end
