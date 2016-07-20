----------------------------------------------------------------------

-- This file implements an autoencoder (diabolo network).
-- It is supposed to be used for CNN unsupervised layer per layer 
-- learning.

----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers

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
	cmd:option('-use_gpu', false, 'wether use the GPU implementation or not')
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
print '==> constructing the network'

encoder = nn.Sequential()
encoder:add(nn.Linear(inputsize,filters))
encoder:add(nn.Sigmoid())

local decoder = nn.Linear(filters,inputsize)
local ae = nn.Sequential()
ae:add(nn.Dropout(opt.drop))
ae:add(encoder)
ae:add(decoder)
ae:add(nn.Sigmoid())
if opt.use_gpu then -- convert to GPU object
	ae = ae:cuda()
end
----------------------------------------------------------------------
-- training
print '==>beginning the training'

file = io.open("error1.txt", "w") -- clear the error1.txt file
file:close()
local criterion = nn.MSECriterion()
if opt.use_gpu then -- convert to GPU object
	criterion = criterion:cuda()
end
local params, gparams = ae:getParameters()

local nsamples = dataset:size(1)*(dataset:size(2) - inputsize + 1)

for k = 1, opt.liter do 
	print('step '..k)
	avgerror = 0 -- average error on the dataset
	acc = 0 -- accumulator for manual batching
	ae:zeroGradParameters()
	for i = 1, dataset:size(1) do
		for j = 1, dataset:size(2) - inputsize + 1 do
			acc = acc + 1
			local temp = dataset[i]:narrow(1,j,inputsize)
			criterion:forward(ae:forward(temp), temp)
			avgerror = avgerror + math.abs(criterion.output)
			ae:backward(temp, criterion:backward(ae.output, temp))
			ae:accUpdateGradParameters(temp, criterion:backward(ae.output, temp), lrate)
			if acc == opt.batch_size then -- batch complete -> updating
				ae:updateParameters(lrate)
				ae:zeroGradParameters()
				acc = 0
			end
		end
	end
	ae:updateParameters(lrate)
	avgerror = avgerror/(nsamples)
	file = io.open("error1.txt", "a") --append the current line
	file:write(tostring(avgerror) .. " " .. tostring(lrate) .. "\n")
	file:close()
	if lrate > opt.mlrate then lrate = lrate/10 end
	--weight decay
	params = params*(1 - opt.wdecay)
	--saving the current nn
	if opt.use_gpu then
		--create a copy of Doubletensor type and save it
		dummy = ae:clone()
		dummy = nn.utils.recursiveType(dummy, 'torch.DoubleTensor')
		torch.save('ae_'..k..'.data', dummy, ascii)
	else
		torch.save('ae_'..k..'.data', ae, ascii)
	end
	if k > 1 then
		os.remove("ae_"..(k-1)..".data")
	end
end

