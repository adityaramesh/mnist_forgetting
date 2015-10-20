function mask_on_first_task(model, batch)
	if batch.targets[1] >= 6 then return end
	assert(batch.targets[1] >= 1 and batch.targets[1] <= 5)

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

		grad_params[1][{{size / 2 + 1, size}}]:zero()
		grad_params[2][{{size / 2 + 1, size}}]:zero()
	end
end

-- XXX: This strategy can't be used with gradient summation, because there is no
-- way to mask half the entries of the second gradient once they have been added
-- to the entries of the first.
function mask_on_both_tasks(model, batch)
	local dataset
	if batch.targets[1] <= 5 then
		dataset = 1
	else
		dataset = 2
	end

	assert(dataset == 1 or dataset == 2)
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
