----------------------------------------------------------------------
--[[This file is supposed to compare classification results, using
the model already built. We will use a fully connected nn of 3 layers,
and try to learn with NLLcriterion and classic MSE criterion.]]
----------------------------------------------------------------------
require 'torch'
require 'nn'
----------------------------------------------------------------------
if not opt then
	print 'MAIN: processing options'
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Options:')
	cmd:option('-directory', 'Brazil', 'selects the directory to use')
	cmd:option('-p', 0.2, 'probability of dropout')
	cmd:option('-lrate', 0.001, 'learning rate')
	cmd:option('-lrate2', 0.00001, 'learning rate for classification')
	cmd:option('-liter', 10, 'number of learning iteration')
	cmd:option('-batch_size', 10, 'size of learning batch')
	cmd:option('-small', false, 'set the supervised training set')
	cmd:text()
end
opt = cmd:parse( arg or {})

class_number = 5 --for now, all the data have 5 classes
lfs.chdir('/'..opt.directory) --change the working directory to the right one
----------------------------------------------------------------------
--charging .data 
if opt.small and opt.directory == 'Lozère' then
	--this is the small dataset construction
	small_data = torch.load('small.data', ascii)
	small_target = torch.load('small_target.data', ascii)
	pre_nn = torch.load('model.data', ascii)
	local temp = pre_nn:forward(small_data[1])
	small_dataset =torch.Tensor(small_data:size(1), temp:size(1)*temp:size(2))
	for k = 1, small_dataset:size(1) do
		small_dataset[k] = pre_nn:forward(small_data[k]):reshape(temp:size(1)*temp:size(2))
	end
end

--complete dataset
dataset = torch.load('final_data.data', ascii)
target = torch.load('target.data', ascii) -- TODO has to be defined for Brazil

print 'MAIN: constructing the final nn'
clasNLL = nn.Sequential()
local temp =  dataset[1]
local input_size = temp:size(1)
local output_size = math.min(input_size, 50)
clasNLL:add(nn.Reshape(input_size))
--first level
l1 = nn.Sequential()
l1:add(nn.Dropout(opt.p))
l1:add(nn.Linear(input_size,output_size))
l1:add(nn.Sigmoid())
clasNLL:add(l1)
--second level
l2 = nn.Sequential()
l2:add(nn.Dropout(opt.p))
l2:add(nn.Linear(output_size,output_size))
l2:add(nn.Sigmoid())
clasNLL:add(l2)

--final softmax classification
clasNLL:add(nn.Linear(output_size,class_number))
clasNLL:add(nn.SoftMax()) -- so the output is a probability vector


--training
print 'MAIN: training the final nn'

if opt.small and opt.directory == 'Lozère' then
	local NLL = nn.CrossEntropyCriterion(torch.mul(torch.Tensor({1/6333, 1/276, 1/80, 1/23, 1/45}),6757))
else
	local NLL = nn.CrossEntropyCriterion()
end
local MSE = nn.MSECriterion()

--unsupervised pre-training as autoencoder
print ' ==> pretraining: level 1'
ae1 = nn.Sequential()
ae1:add(l1)
ae1:add(nn.Linear(output_size, input_size))
ae1:add(nn.Sigmoid())
for k = 1, opt.liter do 
		avgerror = 0 -- average error on the dataset
		acc = 0 -- accumulator for manual batching
		ae1:zeroGradParameters()
	for i = 1, dataset:size(1) do
		acc = acc + 1
		MSE:forward(ae1:forward(dataset[i]), dataset[i])
		avgerror = avgerror + math.abs(MSE.output)
		ae1:backward(dataset[i], MSE:backward(ae1.output, dataset[i]))
		if acc == opt.batch_size then -- batch complete -> updating
			ae1:updateParameters(opt.lrate)
			ae1:zeroGradParameters()
			acc = 0
		end
	end
	ae1:updateParameters(opt.lrate)
	avgerror = avgerror/(dataset:size(1))
	print('average error for pass ', k, ' :', avgerror )
end

l1:evaluate()
--constructing l2 dataset
l2dataset = torch.Tensor(dataset:size(1), output_size)
for k = 1, dataset:size(1) do
	l2dataset[k] = l1:forward(dataset[k])
end
l1:training()

print ' ==> pretraining: level 2'
for k = 1, opt.liter do 
		avgerror = 0 -- average error on the dataset
		acc = 0 -- accumulator for manual batching
		l2:zeroGradParameters()
	for i = 1, l2dataset:size(1) do
		acc = acc + 1
		MSE:forward(l2:forward(l2dataset[i]), l2dataset[i])
		avgerror = avgerror + math.abs(MSE.output)
		l2:backward(l2dataset[i], MSE:backward(l2.output, l2dataset[i]))
		if acc == opt.batch_size then -- batch complete -> updating
			l2:updateParameters(opt.lrate)
			l2:zeroGradParameters()
			acc = 0
		end
	end
	l2:updateParameters(opt.lrate)
	avgerror = avgerror/(l2dataset:size(1))
	print('average error for pass ', k, ' :', avgerror )
end

clasMSE = clasNLL:clone()

before, grad_before = clasNLL:getParameters()
before = before:clone()
grad_before = grad_before:clone()

--supervised final training
print '==>final training'

function argmax_1D(v) -- local
   local length = v:size(1)
   assert(length > 0)

   -- examine on average half the entries
   local maxValue = torch.max(v)
   for i = 1, v:size(1) do
      if v[i] == maxValue then
         return i
      end
   end
end

if opt.small and opt.directory == 'Lozère' then
	for k = 1, opt.liter do 
		print('step ', k)
		acc = 0 -- accumulator for manual batching
		clasNLL:zeroGradParameters()
		clasMSE:zeroGradParameters()
		for i = 1, small_dataset:size(1) do
			acc = acc + 1
			NLL:forward(clasNLL:forward(small_dataset[i]), argmax_1D(small_target[i]))
			MSE:forward(clasMSE:forward(small_dataset[i]), small_target[i])
			--avgerror = avgerror + math.abs(NLL.output)
			clasNLL:backward(small_dataset[i], NLL:backward(clasNLL.output, argmax_1D(small_target[i])))
			clasMSE:backward(small_dataset[i], MSE:backward(clasMSE.output, small_target[i]))
			if acc == opt.batch_size then -- batch complete -> updating
				clasNLL:updateParameters(opt.lrate2)
				clasNLL:zeroGradParameters()
				clasMSE:updateParameters(opt.lrate2)
				clasMSE:zeroGradParameters()
				acc = 0
			end
		end
		clasNLL:updateParameters(opt.lrate2)
		clasMSE:updateParameters(opt.lrate2)
	end
else
	for k = 1, opt.liter do 
		print('step ', k)
		acc = 0 -- accumulator for manual batching
		clasNLL:zeroGradParameters()
		clasMSE:zeroGradParameters()
		for i = 1, dataset:size(1) do
			acc = acc + 1
			NLL:forward(clasNLL:forward(dataset[i]), argmax_1D(target[i]))
			MSE:forward(clasMSE:forward(dataset[i]), target[i])
			--avgerror = avgerror + math.abs(NLL.output)
			clasNLL:backward(dataset[i], NLL:backward(clasNLL.output, argmax_1D(target[i])))
			clasMSE:backward(dataset[i], MSE:backward(clasMSE.output, target[i]))
			if acc == opt.batch_size then -- batch complete -> updating
				clasNLL:updateParameters(opt.lrate2)
				clasNLL:zeroGradParameters()
				clasMSE:updateParameters(opt.lrate2)
				clasMSE:zeroGradParameters()
				acc = 0
			end
		end
		clasNLL:updateParameters(opt.lrate2)
		clasMSE:updateParameters(opt.lrate2)
	end
end
clasMSE:evaluate()
clasNLL:evaluate()

after, grad_after = clasNLL:getParameters()

--[[ does not fit for Brazil
function quality(model)
	count = 0
	for k = 1, dataset:size(1) do
		if argmax_1D(model:forward(dataset[k])) == argmax_1D(target[k]) then
			count = count + 1
		end
	end
	return count/dataset:size(1)
end
]]