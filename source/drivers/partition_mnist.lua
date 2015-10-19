require "torch"
require "image"

local train_fp                 = arg[1]
local test_fp                  = arg[2]
local train_task_1_fp          = arg[3]
local train_task_2_fp          = arg[4]
local train_task_1_reind_fp    = arg[5]
local train_task_2_reind_fp    = arg[6]
local test_task_2_fp           = arg[7]
local test_task_2_reind_fp     = arg[8]

local train_data = torch.load(train_fp)
local test_data  = torch.load(test_fp)

local train_count = train.inputs:size(1)
local width       = train.inputs:size(2)
local test_count  = test.inputs:size(1)
assert(width == test.inputs:size(2))

local train_task_1_ind = torch.le(train.targets, 5)
local train_task_2_ind = torch.ge(train.targets, 6)
local test_task_1_ind  = torch.le(test.targets, 5)
local test_task_2_ind  = torch.ge(test.targets, 6)

local train_task_1_count = train_task_1_ind:sum()
local train_task_2_count = train_task_2_ind:sum()
local test_task_1_count  = test_task_1_ind:sum()
local test_task_2_count  = test_task_2_ind:sum()
assert(train_task_1_count + train_task_2_count == train_count)
assert(test_task_1_count + test_task_2_count == test_count)

print("" .. train_task_1_count .. " train digits from 0--4.")
print("" .. train_task_2_count .. " train digits from 5--9.")
print("" .. test_task_1_count .. " test digits from 0--4.")
print("" .. test_task_2_count .. " test digits from 5--9.")

print("Partitioning training images.")
local train_task_1_images = torch.FloatTensor(train_task_1_count, width, width)
local train_task_2_images = torch.FloatTensor(train_task_2_count, width, width)
local train_task_1_index = 0
local train_task_2_index = 0

for i = 1, train_count do
	if train.targets[i] <= 5 then
		train_task_1_index = train_task_1_index + 1
		train_task_1_images[train_task_1_index]:copy(train.inputs[i])
	else
		train_task_2_index = train_task_2_index + 2
		train_task_2_images[train_task_2_index]:copy(train.inputs[i])
	end
end

assert(train_task_1_index == train_task_1_count)
assert(train_task_2_index == train_task_2_count)
assert(train_task_1_index + train_task_2_index == train_count)

print("Partitioning testing images.")
local test_task_2_images  = torch.FloatTensor(test_task_2_count, width, width)
local test_index = 0

for i = 1, test_count do
	if test.targets[i] >= 6 then
		test_index = test_index + 1
		test_task_2_images[test_index]:copy(test.inputs[i])
	end
end

assert(test_index == test_task_2_count)

local train_task_1_labels = train.targets[train_task_1_ind]
local train_task_2_labels = train.targets[train_task_2_ind]
local test_task_2_labels  = test.targets[test_task_2_ind]

torch.save(train_task_1_fp, {
	inputs  = train_task_1_images,
	targets = train_task_1_labels,
	classes = 10
})

torch.save(train_task_2_fp, {
	inputs  = train_task_2_images,
	targets = train_task_2_labels,
	classes = 10
})

torch.save(test_task_2_fp, {
	inputs  = test_task_2_images,
	targets = test_task_2_labels,
	classes = 10
})

torch.save(train_task_1_reind_fp, {
	inputs  = train_task_1_images,
	targets = train_task_1_labels,
	classes = 5
})

torch.save(train_task_2_reind_fp, {
	inputs  = train_task_2_images,
	targets = train_task_2_labels:add(-5),
	classes = 5
})

torch.save(test_task_2_reind_fp, {
	inputs  = test_task_2_images,
	targets = test_task_2_labels:add(-5),
	classes = 5
})
