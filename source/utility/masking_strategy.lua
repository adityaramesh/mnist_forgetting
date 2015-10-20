function mask_on_both_tasks()
	return function(model, batch)
		local dataset
		if batch.targets[1] <= 5 then
			dataset = 1
		else
			dataset = 2
		end

		-- XXX
		if dataset == 2 then return end
		assert(dataset == 1)
		--assert(dataset == 1 or dataset == 2)
		local fc_layers = model.model:findModules("nn.Linear")

		for i, layer in pairs(fc_layers) do
			if i == #fc_layers then break end
			local _, grad_params = layer:parameters()

			-- There should be one entry for the weights, and one
			-- for the biases.
			assert(#grad_params == 2)
			assert(grad_params[1]:nDimension() == 2)
			assert(grad_params[2]:nDimension() == 1)

			local size = grad_params[1]:size(1)
			assert(size % 2 == 0)
			assert(grad_params[2]:size(1) == size)

			if dataset == 1 then
				grad_params[1][{{size / 2 + 1, size}}]:zero()
				grad_params[2][{{size / 2 + 1, size}}]:zero()
			elseif dataset == 2 then
				grad_params[1][{{1, size / 2}}]:zero()
				grad_params[2][{{1, size / 2}}]:zero()
			end
		end
	end
end

--function mask_on_both_tasks()
--	return function(model, batch)
--		local dataset
--		if batch.targets[1] <= 5 then
--			dataset = 1
--		else
--			dataset = 2
--		end
--
--		if dataset == 2 then return end
--		assert(dataset == 1)
--
--		local fc_layers = model.model:findModules("nn.Linear")
--
--		for i, layer in pairs(fc_layers) do
--			--if i == #fc_layers then break end
--
--			local _, grad_params = layer:parameters()
--
--			-- There should be one entry for the weights, and one
--			-- for the biases.
--			assert(#grad_params == 2)
--			assert(grad_params[1]:nDimension() == 2)
--			assert(grad_params[2]:nDimension() == 1)
--
--			local size = grad_params[1]:size(1)
--			assert(size % 2 == 0)
--			assert(grad_params[2]:size(1) == size)
--
--			if dataset == 1 then
--				grad_params[1][{{size / 2 + 1, size}}]:zero()
--				grad_params[2][{{size / 2 + 1, size}}]:zero()
--			elseif dataset == 2 then
--				grad_params[1][{{1, size / 2}}]:zero()
--				grad_params[2][{{1, size / 2}}]:zero()
--			end
--		end
--	end
--end

function mask_on_first_task()
	return function(model, batch)
		if batch.targets[1] <= 5 then return end

		assert(batch.targets[1] >= 6)
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

			grad_params[1][{{size / 2 + 1, size}}]:zero()
			grad_params[2][{{size / 2 + 1, size}}]:zero()
		end
	end
end
