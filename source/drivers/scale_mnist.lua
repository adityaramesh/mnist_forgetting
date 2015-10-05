require "torch"

input_fp = arg[1]
output_fp = arg[2]

data = torch.load(input_fp)
data.inputs = data.inputs:float():div(255):mul(2):add(-1)
torch.save(output_fp, data)
