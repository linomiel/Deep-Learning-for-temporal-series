----------------------------------------------------------------------

-- This file implements an autoencoder (diabolo network).
-- It is supposed to be used for CNN unsupervised layer per layer 
-- learning.

----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
--require 'unsup'   
--require 'optim'   -- for training in an usual way.

----------------------------------------------------------------------
-- parse command line arguments
if not opt then
	print '==> processing options'
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Options:')
	cmd:option('-k_1', 5,'defines the kernel of the first convolution, i.e; input nodes')
	cmd:option('-nfilter',3,'defines the number of filters, i.e. output nodes')
	cmd:option('-beta', 1, 'prediction error coefficient')
	cmd:option('-liter', 10, 'number of learning iteration on dataset')
	cmd:option('-batch_size', 200, 'size of data groups')
	cmd:option('-ilrate', 0.001, 'initial learning rate')
	cmd:option('-mlrate', 1e-12, 'minimal learning rate')
	cmd:option('-drop', 0.3, 'probability of dropout when training the autoencoders')
	cmd:option('-wdecay', 0, 'set the weight decay epsilon.')
	cmd:text()
end

local opt = cmd:parse( arg or {})
----------------------------------------------------------------------
-- define the parameters
local inputsize = opt.k_1
local filters = opt.nfilter
if filters <= 0 or inputsize <=0 then
	error('there must be nodes in each layer')
end
local lrate = opt.ilrate
----------------------------------------------------------------------
-- construct the diabolo network wanted
print '==> constructiong the network'

encoder = nn.Sequential()
encoder:add(nn.Linear(inputsize,filters))
encoder:add(nn.Sigmoid())

local decoder = nn.Linear(filters,inputsize)
--encoder:add(nn.Diag(filters)) -- not sure if it is usefull or not [for unsup implementation]
--ae = unsup.AutoEncoder(encoder, decoder, opt.beta)  --[for unsup implementation]
local ae = nn.Sequential()
ae:add(nn.Dropout(opt.drop))
ae:add(encoder)
ae:add(decoder)

----------------------------------------------------------------------
--preparing the data
print '==>preprocessing data'
length = dataset:size(2)
learning_data = torch.Tensor(1, inputsize)
for k = 1, dataset:size(1) do
	for i = 1, length - inputsize + 1 do
		learning_data = torch.cat(learning_data, dataset[k]:narrow(1,i,inputsize):reshape(1,inputsize), 1)
	end
end
-- training
print '==>beginning the training'
file = io.open("error1.txt", "w")
local criterion = nn.MSECriterion()
local params, gparams = ae:getParameters()
for k = 1, opt.liter do 
	avgerror = 0 -- average error on the dataset
	acc = 0 -- accumulator for manual batching
	ae:zeroGradParameters()
	for i = 1, learning_data:size(1) do
		acc = acc + 1
		criterion:forward(ae:forward(learning_data[i]), learning_data[i])
		avgerror = avgerror + math.abs(criterion.output)
		ae:backward(learning_data[i], criterion:backward(ae.output, learning_data[i]))
		if acc == opt.batch_size then -- batch complete -> updating
			ae:updateParameters(lrate)
			ae:zeroGradParameters()
			acc = 0
		end
	end
	ae:updateParameters(lrate)
	avgerror = avgerror/(learning_data:size(1))
	file:write(tostring(avgerror) .. " " .. tostring(lrate) .. "\n")
	if lrate > opt.mlrate then lrate = lrate/10 end
	--weight decay
	params = params*(1 - opt.wdecay)
end
file:close()
--[[ optim implemntation 
local params, gradParams = ae:getParameters()
local optimState = {learningRate = 0.01}


-- a function used by svg. -- can't compile
local function feval(params,i)
	gradParams:zero()
	local loss = criterion:forward(ae:forward(learning_data[i]),learning_data[i])
	avgerror = avgerror + loss --accumulating
	ae:backward(learning_data[i],criterion:backward(ae.output, learning_data[i]))
	return loss,gradParams
end 

for k = 1, opt.liter do
	avgerror = 0
	for i = 1, learning_data:size(1) do
		optim.sgd(feval, params, optimState)
	end
	avgerror = avgerror/learning_data:size(1)
	print('average error on pass ', k, ' :', avgerror)
end ]]