----------------------------------------------------------------------

-- This file contain the nn in charge of feature extraction for each pixel.
-- His structure is :
-- conv -> squashing -> pooling -> conv -> squashing -> pooling

----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'lfs'	  -- allows changin the current directory
----------------------------------------------------------------------

-- parse command line arguments
if not opt then
	print 'MAIN: processing options'
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Options:')
	cmd:option('-directory', 'Brazil', 'selects the directory to use')
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
	cmd:text()
end

opt = cmd:parse( arg or {})


class_number = 5 --for now, all the data have 5 classes
here = lfs.currentdir()
if not lfs.chdir(here..'/'..opt.directory) then--change the working directory to the dataset directory
	error('fail in directory change')
end
dataset = torch.load('dataset.data')
input = dataset:size(2)
----------------------------------------------------------------------
print 'MAIN: Constructing the model'
--just the first convolution/pooling layer for now.
model = nn.Sequential()
first_layer = nn.Sequential()
second_layer = nn.Sequential()
conv1 = nn.TemporalConvolution(1, opt.nfilter, opt.k_1)
conv2 = nn.TemporalConvolution(opt.nfilter, opt.nfilter2, opt.k_2)

first_layer:add(nn.Reshape(input, 1))
first_layer:add(conv1)
first_layer:add(nn.Sigmoid())
first_layer:add(nn.TemporalMaxPooling(opt.p_1))

local pre_pool_len = input - opt.k_1 + 1
local frame_length = math.floor((pre_pool_len - opt.p_1)/opt.p_1 + 1) -- size of the feature series

second_layer:add(conv2)
second_layer:add(nn.Sigmoid())
second_layer:add(nn.TemporalMaxPooling(opt.p_2))

model:add(first_layer)
model:add(second_layer)
----------------------------------------------------------------------
if opt.save ~= "" then
	--creates and and go in the experience saving directory
	lfs.mkdir(opt.save)
	here = lfs.currentdir()
	lfs.chdir(here..'/'..opt.save)
	file = io.open("params.txt", "w")
	for key, value in pairs(opt) do
		file:write( key .." : " .. tostring(value) .. "\n")
	end
	file:close()
end
-- creates and train the first encoder.

print'MAIN: first convolution external training'
dofile '../../autoencoder.lua'

print'MAIN: copying the first convolution parameters'
params, grad_params = encoder:getParameters()
cparams, cgrad_params = conv1:getParameters()
for k = 1, params:size(1) do
	cparams[k] = params[k]
	cgrad_params[k] = grad_params[k]
end

----------------------------------------------------------------------
-- creates and train the second encoder, after preprocessing

print 'MAIN: construction of the second dataset'
--we will now construct the dataset for the second layer external training
dataset2 = torch.Tensor(dataset:size(1),frame_length*opt.nfilter)
--creating
for k = 1, dataset:size(1) do
	dataset2[k] = (first_layer:forward(dataset[k])):reshape(frame_length*opt.nfilter)
end
--redimensioning
input_conv2 = (opt.nfilter*opt.k_2) --number of entry nodes in layer 2
learning_data2 = torch.Tensor(1, input_conv2)
for k = 1, dataset2:size(1) do
	for i = 1, dataset:size(2) - input_conv2 + 1 do
		learning_data2 = torch.cat(learning_data2, dataset[k]:narrow(1,i,input_conv2):reshape(1,input_conv2), 1)
	end
end
print 'MAIN: second convolution external training'
dofile '../../autoencoder2.lua'

print'MAIN: copying the second convolution parameters'
params2, grad_params2 = encoder2:getParameters()
cparams2, cgrad_params2 = conv2:getParameters()
for k = 1, params2:size(1) do
	cparams2[k] = params2[k]
	cgrad_params2[k] = grad_params2[k]
end

if opt.save ~= "" then
	print 'MAIN: saving  current data'
	torch.save('model.data', model, ascii) -- saving the model
	--building the forwarded image of the dataset
	local temp = model:forward(dataset[1])
	final_data = torch.Tensor(dataset:size(1), temp:size(1)*temp:size(2))
	for k = 1, dataset:size(1) do
		final_data[k] = (model:forward(dataset[k])):reshape(temp:size(1)*temp:size(2))
	end
	torch.save('final_data.data', final_data, ascii)

	--same thing for each class, and storing it in .txt
	file = io.open("classes.txt", "w")
	for k = 1, class_number do
		class = torch.load('../class'..tostring(k)..'.data', ascii)
		for i = 1, class:size(1) do
			if opt.directory == "Brazil" then -- we avoid the first column, holding patch ID.
				local temp = model:forward(class[i]:narrow(1,2,dataset:size(2))):reshape(temp:size(1)*temp:size(2))
			else
				local temp = model:forward(class[i]):reshape(temp:size(1)*temp:size(2))
			end
			file:write(k .. " ")
			for j = 1, temp:size(1) do
				file:write(tostring(temp[j]), " ")
			end
			file:write("\n")
		end
	end
	file:close()
end