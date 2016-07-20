---------------------------------------------------------------------

-- This file implements the autoencoder wich will be used in the model
-- secon layer.
-- It takes in entry matrices : temporal series of features.

----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers

----------------------------------------------------------------------

if not opt then
	print '==> processing options'
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Options:')
	cmd:option('-k_2', 5,'defines the kernel of the second convolution')
	cmd:option('-nfilter',3,'defines the number of filters for the first convolution')
	cmd:option('-nfilter2',9,'defines the number of filters for the second convolution, i.e. output nodes')
	cmd:option('-beta', 1, 'prediction error coefficient')
	cmd:option('-liter', 10, 'number of learning iteration on dataset')
	cmd:option('-batch_size', 200, 'size of data groups')
	cmd:option('-ilrate', 0.001, 'initial learning rate')
	cmd:option('-mlrate', 1e-12, 'minimal learning rate')
	cmd:option('-drop', 0.3, 'probability of dropout when training the autoencoders')
	cmd:option('-use_gpu', false, 'wether use the GPU implementation or not')
	cmd:text()
end

local opt = cmd:parse( arg or {})
----------------------------------------------------------------------

local inputsize = opt.nfilter*opt.k_2
local filters = opt.nfilter2
if filters <= 0 or inputsize <=0 then
	error('there must be nodes in each layer')
end
local lrate = opt.ilrate
----------------------------------------------------------------------
-- construct the diabolo network wanted (same as the first one)
print '==> constructing the network'

encoder2 = nn.Sequential()
encoder2:add(nn.Linear(inputsize,filters))
encoder2:add(nn.Sigmoid())

local decoder2 = nn.Linear(filters,inputsize)
local ae2 = nn.Sequential()
ae2:add(nn.Dropout(opt.drop))
ae2:add(encoder2)
ae2:add(decoder2)
ae2:add(nn.Sigmoid())
if opt.use_gpu then -- convert to GPU object
	ae2 = ae2:cuda()
end
----------------------------------------------------------------------
--training
print '==>beginning the training'

local criterion = nn.MSECriterion()
if opt.use_gpu then -- convert to GPU object
	criterion = criterion:cuda()
end
file = io.open("error2.txt", "w") -- clear the error2.txt file
file:close()
local params,gparams = ae2:getParameters()
local nsamples = dataset:size(1)*(dataset:size(2) - inputsize + 1) 

for k = 1, opt.liter do
	print('step '..k)
	avgerror = 0 -- average error on the dataset
	acc = 0 -- accumulator for manual batching
	ae2:zeroGradParameters()
	for i = 1, dataset:size(1) do
		local processed = first_layer:forward(dataset[i]):resize(frame_length*opt.nfilter)
		for j = 1, processed:size(1) - inputsize + 1 do
			acc = acc + 1
			temp = processed:narrow(1, j, inputsize)
			criterion:forward(ae2:forward(temp), temp)
			avgerror = avgerror + math.abs(criterion.output)
			ae2:backward(temp, criterion:backward(ae2.output, temp))
			ae2:accUpdateGradParameters(temp, criterion:backward(ae2.output, temp), lrate)
			if acc == opt.batch_size then -- batch complete -> updating
				ae2:updateParameters(lrate)
				ae2:zeroGradParameters()
				acc = 0
			end
		end
	end
	ae2:updateParameters(lrate)
	avgerror = avgerror/(nsamples)
	file = io.open("error2.txt", "a") --append the current line
	file:write(tostring(avgerror) .. " " .. tostring(lrate) .. "\n")
	file:close()
	if lrate > opt.mlrate then lrate = lrate/10 end
	--weight decay
	params = params*(1 - opt.wdecay)
	--saving the current nn
	if opt.use_gpu then
		--create a copy of Doubletensor type and save it
		dummy = ae2:clone()
		dummy = nn.utils.recursiveType(dummy, 'torch.DoubleTensor')
		torch.save('ae2_'..k..'.data', dummy, ascii)
	else
		torch.save('ae2_'..k..'.data', ae2, ascii)
	end
	if k > 1 then
		os.remove("ae_2"..(k-1)..".data")
	end
end
