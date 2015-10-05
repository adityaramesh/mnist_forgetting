require "torch"

input_fp  = arg[1]
left_output_fp = arg[2]
right_output_fp = arg[3]

data = torch.load(input_fp)

-- The labels are actually from 1 to 10, not 0 to 9.
left_ind = torch.le(data.targets, 5)
right_ind = torch.ge(data.targets, 6)

count = data.inputs:size(1)
width = data.inputs:size(2)
left_count = left_ind:sum()
right_count = right_ind:sum()
assert(left_count + right_count == count)

print("" .. left_count .. " digits from 0--4.")
print("" .. right_count .. " digits from 5--9.")

left_labels = data.targets[left_ind]
right_labels = data.targets[right_ind]:add(-5)

left_images = torch.FloatTensor(left_count, width, width)
right_images = torch.FloatTensor(right_count, width, width)

left_idx = 0
right_idx = 0

for i = 1, count do
	if data.targets[i] <= 5 then
		left_idx = left_idx + 1
		left_images[left_idx]:copy(data.inputs[i])
	else
		right_idx = right_idx + 1
		right_images[right_idx]:copy(data.inputs[i])
	end
end

assert(left_idx == left_count)
assert(right_idx == right_count)
assert(left_idx + right_idx == count)

torch.save(left_output_fp, {inputs = left_images, targets = left_labels, classes = 5})
torch.save(right_output_fp, {inputs = right_images, targets = right_labels, classes = 5})
