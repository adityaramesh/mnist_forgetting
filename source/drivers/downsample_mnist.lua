require "torch"
require "image"

input_fp = arg[1]
output_fp = arg[2]
scaling_factor = tonumber(arg[3])

data = torch.load(input_fp)
assert(data.inputs:nDimension() == 3)

count = data.inputs:size(1)
width = data.inputs:size(2)
height = data.inputs:size(3)

assert(data.inputs:type() == "torch.FloatTensor")
assert(width == height)
assert(width % scaling_factor == 0)

new_inputs = torch.FloatTensor(count, width / scaling_factor, height / scaling_factor)

for i = 1, count do
	if i % 1000 == 1 then
		print("Working on image " .. i .. " / " .. count .. ".")
	end
	image.scale(new_inputs[i], data.inputs[i])
end

torch.save(output_fp, {inputs = new_inputs, targets = data.targets, classes = data.classes})
