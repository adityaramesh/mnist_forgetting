require "source/models/cnn_3x3.lua"

local info = get_model_info()
local model = info.model
local p2, g2 = model:get(2):parameters()
local p5, g5 = model:get(5):parameters()
print("Second layer params:")
print(p2)
print("Second layer grad params:")
print(g2)
print("Fifth layer params:")
print(p5)
print("Fifth layer grad params:")
print(g5)

local conv_layers = model:findModules("nn.SpatialConvolutionMM")
for i = 1, #conv_layers do
	local p, g = conv_layers[i]:parameters()
	print("Conv layer " .. i .. " parameters:")
	print(p)
	print("Conv layer " .. i .. " grad parameters:")
	print(g)
	print("Output?")
	print(conv_layers[i].output)
end
