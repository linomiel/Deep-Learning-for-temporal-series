----------------------------------------------------
--This file does the second part of model.lua, for interrupted experiences.

-- creates and train the second encoder
--first, we need to know wich input size is needed
--narrow_size = opt.k_1 - 1 + opt.p_1*opt.k_2/nfilter
----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'lfs'	  -- allows changing the current directory
----------------------------------------------------------------------

-- parse command line arguments
if not opt then
	print 'MAIN: processing options'
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Options:')
	cmd:option('-directory', 'Brazil', 'selects the directory to use')
	cmd:option('-subdirectory', "", 'the subdirectory where the first convolution has been trained')
	cmd:option('-nfilter', 50,'defines the number of filters for the first convolution') --used to be 3 -> not enough data?
	cmd:option('-nfilter2',90,'defines the number of filters for the second convolution')
	cmd:option('-k_1', 5,'defines the kernel of the first convolution')
	cmd:option('-k_2', 5,'defines the kernel of the second convolution')
	cmd:option('-drop', 0.3, 'probability of dropout when training the autoencoders')
	cmd:option('-p_1', 2, 'first pooling size')
	cmd:option('-p_2', 2, 'second pooling size')
	cmd:option('-beta', 1, 'prediction error coefficient')
	cmd:option('-liter', 10, 'number of learning iteration on dataset')
	cmd:option('-batch_size', 1000, 'size of data groups')
	cmd:option('-ilrate', 0.1, 'initial learning rate')
	cmd:option('-mlrate', 1e-12, 'minimal learning rate')
	cmd:option('-save', "", 'wether or not the neural network should be saved.')
	cmd:option('-wdecay', 0, 'set the weight decay epsilon.')
	cmd:option('-use_gpu', false, 'wether use the GPU implementation or not')
	cmd:text()
end


opt = cmd:parse( arg or {})


class_number = 5 --for now, all the data have 5 classes
here = lfs.currentdir()
if not lfs.chdir(here..'/'..opt.directory) then--change the working directory to the dataset directory
	error('fail in directory change')
end
dataset = torch.load('dataset.data')
if opt.use_gpu then
	require 'cutorch' --for GPU implementation
	dataset = dataset:cuda()
end
input = dataset:size(2)
if not lfs.chdir(here..'/'..opt.directory .. '/' ..opt.subdirectory) then--change the working directory to the model directory
	error('fail in subdirectory change')
end
print 'MAIN: Constructing the model'
--just the first convolution/pooling layer for now.
model = nn.Sequential()
first_layer = torch.load('model1.data'):get(1) --the trained first layer
second_layer = nn.Sequential()
conv2 = nn.TemporalConvolution(opt.nfilter, opt.nfilter2, opt.k_2)

local pre_pool_len = input - opt.k_1 + 1
frame_length = math.floor((pre_pool_len - opt.p_1)/opt.p_1 + 1) -- size of the feature series

second_layer:add(conv2)
second_layer:add(nn.Sigmoid())
second_layer:add(nn.TemporalMaxPooling(opt.p_2))

model:add(first_layer) --has been trained
model:add(second_layer) --is yet to be trained

if opt.use_gpu then
	model = model:cuda()
end

lfs.mkdir(opt.save)
if not lfs.chdir(here..'/'..opt.directory .. '/' ..opt.subdirectory .. '/' .. opt.save) then--change the working directory to the final directory
	error('fail in subsubdirectory change')
end
print 'MAIN: second convolution external training'
dofile '../../../autoencoder2.lua'

print'MAIN: copying the second convolution parameters'
params2, grad_params2 = encoder2:getParameters()
cparams2, cgrad_params2 = conv2:getParameters()
for k = 1, params2:size(1) do
	cparams2[k] = params2[k]
	cgrad_params2[k] = grad_params2[k]
end

if opt.save ~= "" then
	print 'MAIN: saving  current data'
	if opt.use_gpu then 
		-- we create a double tensor version of model and save it
		dummy = model:clone()
		dummy = nn.utils.recursiveType(dummy, 'torch.DoubleTensor')
		torch.save('model2.data', dummy, ascii)
	else
		torch.save('model2.data', model, ascii) -- saving the model
	end
end