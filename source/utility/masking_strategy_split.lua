function mask_on_first_task(model, batch)
	if batch.targets[1] >= 6 then return end
	assert(batch.targets[1] >= 1 and batch.targets[1] <= 5)

	for i, layer in pairs(model.model.fg:topsort()) do
		local size = 512
		local module = layer.data.module
		if module and module.gradWeight and module.gradWeight:size(1) == size then
			module.gradWeight[{{size / 2 + 1, size}}]:zero()
			module.gradBias[{{size / 2 + 1, size}}]:zero()
		end
	end
end

function mask_on_second_task(model, batch)
	if batch.targets[1] >= 6 then return end
	assert(batch.targets[1] >= 1 and batch.targets[1] <= 5)

	for i, layer in pairs(model.model.fg:topsort()) do
		local size = 512
		local module = layer.data.module
		if module and module.gradWeight and module.gradWeight:size(1) == size then
			module.gradWeight[{{1, size / 2}}]:zero()
			module.gradBias[{{1, size / 2}}]:zero()
		end
	end
end
