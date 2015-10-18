function mask_on_both_tasks()
	return function(model, batch)
		assert(batch.dataset == 1 or batch.dataset == 2)
		local fc_layers = model.model:findModules("nn.Linear")

		for _, layer in pairs(fc_layers) do
			local _, grad_params = layer:parameters()

			-- There should be one entry for the weights, and one
			-- for the biases.
			assert(#grad_params == 2)
			assert(grad_params[1]:nDimension() == 2)
			assert(grad_params[2]:nDimension() == 1)

			local size = grad_params[1]:size(1)
			assert(size % 2 == 0)
			assert(grad_params[2]:size(1) == size)

			if batch.dataset == 1 then
				grad_params[1][{{1, size / 2}}]:zero()
				grad_params[2][{{1, size / 2}}]:zero()
			elseif batch.dataset == 2 then
				grad_params[1][{{size / 2 + 1, size}}]:zero()
				grad_params[2][{{size / 2 + 1, size}}]:zero()
			end
		end
	end
end

function mask_on_first_task()
	return function(model, batch)
		assert(batch.dataset == 1 or batch.dataset == 2)
		if batch.dataset == 2 then return end

		local fc_layers = model.model:findModules("nn.Linear")

		for _, layer in pairs(fc_layers) do
			local _, grad_params = layer:parameters()

			-- There should be one entry for the weights, and one
			-- for the biases.
			assert(#grad_params == 2)
			assert(grad_params[1]:nDimension() == 2)
			assert(grad_params[2]:nDimension() == 1)

			local size = grad_params[1]:size(1)
			assert(size % 2 == 0)
			assert(grad_params[2]:size(1) == size)

			grad_params[1][{{1, size / 2}}]:zero()
			grad_params[2][{{1, size / 2}}]:zero()
		end
	end
end
