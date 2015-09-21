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
