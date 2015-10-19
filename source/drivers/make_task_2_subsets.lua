require "torch"

local input_fp     = arg[1]
local output_fp    = arg[2]
local is_reindexed = arg[3]

local data  = torch.load(input_fp)
local count = data.inputs:size(1)
local width = data.inputs:size(2)

print("Computing indices of instances in each class.")
local indices = {}
local sum = 0

for i = 1, 5 do
	if is_reindexed == "true" then
		local t = torch.eq(data.targets, i)
		indices[i] = data.targets[t]
		sum = sum + t:sum()
	elseif is_reindexed == "false" then
		local t = torch.eq(data.targets, i + 5)
		indices[i] = data.targets[t]
		sum = sum + t:sum()
	else
		error("Invalid value `" .. is_reindexed .. "` for argument three.")
	end
end

assert(count == sum)

local subset_size = 1000
local classes = 5
local instances_per_class = subset_size / classes

print("Computing indices for data partitions.")
local partition_indices = torch.Tensor(subset_size)

for i = 1, 5 do
	local perm = torch.randperm(indices[i]:size(1))
	for j = 1, instances_per_class do
		partition_indices[(i - 1) * instances_per_class + j] = indices[i][perm[j]]
	end
end

print("Forming subset.")
local perm    = torch.randperm(subset_size)
local images  = torch.FloatTensor(subset_size, width, width)
local targets = torch.IntTensor(subset_size)

for i = 1, subset_size do
	local index = partition_indices[perm[i]]
	images[{{i}}]:copy(data.inputs[{{index}}])
	targets[{{i}}]:copy(data.targets[{{index}}])
end

print("Saving data.")
torch.save(output_fp, {
	inputs  = inputs,
	targets = targets,
	classes = data.classes
})
