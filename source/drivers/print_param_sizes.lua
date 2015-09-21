require "source/models/cnn_3x3.lua"

local info = get_model_info()
local model = info.model
local p2, g2 = model:get(2):parameters()
local p5, g5 = model:get(5):parameters()
print(p2)
print(g2)
print(p5)
print(g5)
