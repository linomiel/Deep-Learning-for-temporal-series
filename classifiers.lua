----------------------------------------------------------------------
--[[This file is supposed to compare classification results, using
the model already built. We will use a fully connected nn of 3 layers,
and try to learn with NLLcriterion and classic MSE criterion.]]
----------------------------------------------------------------------
require 'torch'
require 'nn'
require 'csvigo'
require 'lfs'	  -- allows changing the current directory
----------------------------------------------------------------------
if not opt then
	print 'MAIN: processing options'
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Options:')
	cmd:option('-directory', 'Brazil', 'selects the directory to use')
	cmd:option('-subdir', "", 'the subdirectory considered')
	cmd:option('-save', "", 'the model directory where data will be saved')
	cmd:option('-p', 0.3, 'probability of dropout')
	cmd:option('-ilrate', 0.001, 'initial learning rate')
	cmd:option('-mlrate', 1e-12, 'minimal learning rate, used for classification')
	cmd:option('-liter', 10, 'number of learning iteration')
	cmd:option('-batch_size', 10, 'size of learning batch')
	cmd:text()
end
opt = cmd:parse( arg or {})

class_number = 5 --for now, all the data have 5 classes
here = lfs.currentdir()
if not lfs.chdir(here..'/'..opt.directory..'/'..opt.subdir .. '/' ..opt.save) then--change the working directory to the dataset directory
	error('fail in directory change')
end
----------------------------------------------------------------------
--charging .data 
print 'MAIN: preprocessing the data'
if opt.directory == "Lozère" then
	dataset = torch.load('final_data.data', ascii)
	--creating a training and testing set
	--dataset is also classified data, so we just randomly split it.
	index = torch.randperm(dataset:size(1)):long()
	test = dataset:index(1, index):narrow(1, 1, math.floor(dataset:size(1)/2))
	train = dataset:index(1,index):narrow(1, math.floor(dataset:size(1)/2), dataset:size(1) - math.floor(dataset:size(1)/2))
	target = torch.load('../target.data', ascii) --care!
	test_target = target:index(1, index):narrow(1, 1, math.floor(dataset:size(1)/2))
	train_target = target:index(1,index):narrow(1, math.floor(dataset:size(1)/2), dataset:size(1) - math.floor(dataset:size(1)/2))

elseif opt.directory == "Brazil" then
	--we construct training/testing set by spliting each class by patch.
	--finaldata is not loaded!
	for l = 1, class_number do
		class = torch.load('../../class'..l..'.data', ascii)
		patch_nb = class[class:size(1)][1]
		patch_index = torch.randperm(patch_nb):long()
		acc = 0
		k = 1
		--we now feed patches to training test while it doe not reach half the class.
		while acc < class:size(1)/2 do
			patch = patch_index[k]
			init = 1
			while class[init][1] ~= patch do --locate the patch beginning
				init = init + 1
			end
			range = 0
			while init + range <= class:size(1) and class[init + range][1] == patch do --count patch elements
				range = range + 1
			end
			if pre_train then
				pre_train = torch.cat(1,pre_train,class:narrow(2,2,class:size(2) - 1):narrow(1,init,range))
				new_target = torch.Tensor(range, class_number):zero()
				for i = 1, range do
					new_target[i][l] = 1
				end
				train_target = torch.cat(1,train_target,new_target)
			else -- if they have not been initialized yet.
				pre_train = class:narrow(2,2,class:size(2) - 1):narrow(1,init,range):clone()
				train_target = torch.Tensor(range, class_number):zero()
				for i = 1, range do
					train_target[i][l] = 1
				end
			end
			k = k + 1
			acc = acc + range
		end
		--and then we use the remaining patches for test
		for k = k, patch_index:size(1) do
			patch = patch_index[k]
			init = 1
			while class[init][1] ~= patch do
				init = init + 1
			end
			range = 0
			while init + range <= class:size(1) and class[init + range][1] == patch do
				range = range + 1
			end
			if pre_test then
				pre_test = torch.cat(1,pre_test,class:narrow(2,2,class:size(2) - 1):narrow(1,init,range))
				new_target = torch.Tensor(range, class_number):zero()
				for i = 1, range do
					new_target[i][l] = 1
				end
				test_target = torch.cat(1,test_target,new_target)
			else -- if they have not been initialized yet.
				pre_test = class:narrow(2,2,class:size(2) - 1):narrow(1,init,range):clone()
				test_target = torch.Tensor(range, class_number):zero()
				for i = 1, range do
					test_target[i][l] = 1
				end
			end
		end
	end
	--we preprocess training and testing set
	pre_nn = torch.load('model2.lua', ascii)
	temp = pre_nn:forward(pre_train[1])
	local size = temp:size(1)*temp:size(2)
	train = torch.Tensor(pre_train:size(1), size)
	for k = 1, train:size(1) do
		train[k] = pre_nn:forward(pre_train[k]):reshape(size)
	end
	test = torch.Tensor(pre_test:size(1), size)
	for k = 1, test:size(1) do
		test[k] = pre_nn:forward(pre_test[k]):reshape(size)
	end
	--train and test are created, we will now create a dataset for AE pretraining: test + train?
	dataset = torch.cat(test,train,1)
	dataset = dataset:index(1, torch.randperm(dataset:size(1)):long())
else
	error("unknown directory")
end

print 'MAIN: constructing the final nn'
classNLL = nn.Sequential()
local temp =  dataset[1]
local input_size = temp:size(1)
local output_size = math.min(input_size, 50)
classNLL:add(nn.Reshape(input_size))
--first level
l1 = nn.Sequential()
l1:add(nn.Dropout(opt.p))
l1:add(nn.Linear(input_size,output_size))
l1:add(nn.Sigmoid())
classNLL:add(l1)
--second level
l2 = nn.Sequential()
l2:add(nn.Dropout(opt.p))
l2:add(nn.Linear(output_size,output_size))
l2:add(nn.Sigmoid())
classNLL:add(l2)

--final softmax classification
classNLL:add(nn.Linear(output_size,class_number))
classNLL:add(nn.SoftMax()) -- so the output is a probability vector


--training
print 'MAIN: training the final nn'

local NLL = nn.CrossEntropyCriterion()
local MSE = nn.MSECriterion()

--unsupervised pre-training as autoencoder
print ' ==> pretraining: level 1'
ae1 = nn.Sequential()
ae1:add(l1)
ae1:add(nn.Linear(output_size, input_size))
ae1:add(nn.Sigmoid())

lrate = opt.ilrate
file = io.open("class_error1.txt", "w")
for k = 1, opt.liter do 
		avgerror = 0 -- average error on the dataset
		acc = 0 -- accumulator for manual batching
		ae1:zeroGradParameters()
	for i = 1, dataset:size(1) do
		acc = acc + 1
		MSE:forward(ae1:forward(dataset[i]), dataset[i])
		avgerror = avgerror + math.abs(MSE.output)
		ae1:backward(dataset[i], MSE:backward(ae1.output, dataset[i]))
		ae1:accUpdateGradParameters(dataset[i], MSE:backward(ae1.output, dataset[i]), lrate)
		if acc == opt.batch_size then -- batch complete -> updating
			ae1:updateParameters(lrate)
			ae1:zeroGradParameters()
			acc = 0
		end
	end
	ae1:updateParameters(lrate)
	avgerror = avgerror/(dataset:size(1))
	file:write(tostring(avgerror) .. " " .. tostring(lrate) .. "\n")
	if lrate > opt.mlrate then
		lrate = lrate/10
	end
end
file:close()

l1:evaluate()
--constructing l2 dataset
l2dataset = torch.Tensor(dataset:size(1), output_size)
for k = 1, dataset:size(1) do
	l2dataset[k] = l1:forward(dataset[k])
end
l1:training()

print ' ==> pretraining: level 2'

lrate = opt.ilrate
file = io.open("class_error2.txt", "w")
for k = 1, opt.liter do 
		avgerror = 0 -- average error on the dataset
		acc = 0 -- accumulator for manual batching
		l2:zeroGradParameters()
	for i = 1, l2dataset:size(1) do
		acc = acc + 1
		MSE:forward(l2:forward(l2dataset[i]), l2dataset[i])
		avgerror = avgerror + math.abs(MSE.output)
		l2:backward(l2dataset[i], MSE:backward(l2.output, l2dataset[i]))
		l2:accUpdateGradParameters(l2dataset[i], MSE:backward(l2.output, l2dataset[i]), lrate)
		if acc == opt.batch_size then -- batch complete -> updating
			l2:updateParameters(lrate)
			l2:zeroGradParameters()
			acc = 0
		end
	end
	l2:updateParameters(lrate)
	avgerror = avgerror/(l2dataset:size(1))
	file:write(tostring(avgerror) .. " " .. tostring(lrate) .. "\n")
	if lrate > opt.mlrate then
		lrate = lrate/10
	end
end
file:close()

classMSE = classNLL:clone()

before, grad_before = classNLL:getParameters()
before = before:clone()
grad_before = grad_before:clone()
torch.save('weight_before.data', before, ascii)
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

for k = 1, opt.liter do 
	print('step ', k)
	acc = 0 -- accumulator for manual batching
	classNLL:zeroGradParameters()
	classMSE:zeroGradParameters()
	for i = 1, train:size(1) do
		acc = acc + 1
		NLL:forward(classNLL:forward(train[i]), argmax_1D(train_target[i]))
		MSE:forward(classMSE:forward(train[i]), train_target[i])
		--avgerror = avgerror + math.abs(NLL.output)
		classNLL:backward(train[i], NLL:backward(classNLL.output, argmax_1D(train_target[i])))
		classNLL:accUpdateGradParameters(dataset[i], NLL:backward(classNLL.output, train_target[i]), lrate)
		classMSE:backward(train[i], MSE:backward(classMSE.output, train_target[i]))
		classMSE:accUpdateGradParameters(dataset[i], MSE:backward(classMSE.output, train_target[i]), lrate)
		if acc == opt.batch_size then -- batch complete -> updating
			classNLL:updateParameters(opt.mlrate)
			classNLL:zeroGradParameters()
			classMSE:updateParameters(opt.mlrate)
			classMSE:zeroGradParameters()
			acc = 0
		end
	end
	classNLL:updateParameters(opt.mlrate)
	classMSE:updateParameters(opt.mlrate)
end

classMSE:evaluate()
classNLL:evaluate()
torch.save('classNLL.data', classNLL, ascii)
torch.save('classMSE.data', classMSE, ascii)

--testing
MSEconfusion = torch.Tensor(class_number, class_number):zero()
NLLconfusion = torch.Tensor(class_number, class_number):zero()

for k = 1, test:size(1) do
	x = argmax_1D(test_target[k])
	y_MSE = argmax_1D(classMSE:forward(test[k]))
	y_NLL = argmax_1D(classNLL:forward(test[k]))
	MSEconfusion[x][y_MSE] = MSEconfusion[x][y_MSE] + 1
	NLLconfusion[x][y_NLL] = NLLconfusion[x][y_NLL] + 1
end
torch.save('MSEconfusion.data', MSEconfusion, ascii)
torch.save('NLLconfusion.data', NLLconfusion, ascii)

--computes the kappas
s0_MSE = 0
se_MSE = 0
s0_NLL = 0
se_NLL = 0
for k = 1, class_number do
	s0_MSE = s0_MSE + MSEconfusion[k][k]
	s0_NLL = s0_NLL + NLLconfusion[k][k]
	tempMSE = 0
	tempMSE2 = 0
	tempNLL = 0
	tempNLL2 = 0
	for i = 1, class_number do
		tempNLL2 = tempNLL2 + NLLconfusion[k][i]
		tempNLL = tempNLL + NLLconfusion[i][k]
		tempMSE2 = tempMSE2 + MSEconfusion[k][i]
		tempMSE = tempMSE + MSEconfusion[i][k]
	end
	se_MSE = se_MSE + tempMSE*tempMSE2
	se_NLL = se_NLL + tempNLL*tempNLL2
end
p0_MSE = s0_MSE/test:size(1)
p0_NLL = se_NLL/test:size(1)
pe_MSE = se_MSE/(test:size(1)*test:size(1))
pe_NLL = se_NLL/(test:size(1)*test:size(1))

kappa_MSE = (p0_MSE - pe_MSE)/(1 - pe_MSE)
kappa_NLL = (p0_NLL - pe_NLL)/(1 - pe_NLL)

file=io.open("ANN_kappa.txt",'w')
file:write("kappa MSE: "..kappa_MSE.."\n")
file:write("kappa NLL: "..kappa_NLL.."\n")
file:close()