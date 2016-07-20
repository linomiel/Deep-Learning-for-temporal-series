----------------------------------------------------------------------
--[[This file handle the data from MatoGrosso.
As did ae_data, it is supposed to create :
 - dataset, wich is the general dataset from complete image. [unsup]
 - one class data per class. [sup]
 ]]
 ----------------------------------------------------------------------
require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'csvigo'  -- For CSV format loading
----------------------------------------------------------------------
if not opt then
	print '==> processing options'
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Options:')
	cmd:option('-s', false, 'if supervised training data is to be computed and stored (classes and dataset)')
	cmd:option('-u', false, 'if unsupervised training data is to be computed and stored (learning_data)')
	cmd:text()
end
opt = cmd:parse( arg or {})
----------------------------------------------------------------------
--charging data
if opt.s then
	class_data = {}
	--those have parcel numbers
	class_data[1] = csvigo.load{path = '../Données et R/Cotton.txt', separator = ' ', header = false, mode = 'raw'}
	class_data[2] = csvigo.load{path = '../Données et R/Soybean + Cotton.txt', separator = ' ', header = false, mode = 'raw'}
	class_data[3] = csvigo.load{path = '../Données et R/Soybean + Cover.txt', separator = ' ', header = false, mode = 'raw'}
	class_data[4] = csvigo.load{path = '../Données et R/Soybean + Maize.txt', separator = ' ', header = false, mode = 'raw'}
	class_data[5] = csvigo.load{path = '../Données et R/Soybean.txt', separator = ' ', header = false, mode = 'raw'}

	class = {}
	for k = 1, 5 do
		class[k] = torch.Tensor(class_data[k])
	end
	--data normalization (common to all classes)
	local mean = 0
	local sum = 0 --number of pixel in all classes
	for k = 1, 5 do
		sum = sum + #class_data[k]
	end
	--computes mean
	for k = 1, 5 do
		for j = 1, class[k]:size(1) do
			for i = 2, class[k]:size(2) do -- we avoid the fist column,where the patch id is stored.
				mean = class[k][j][i] + mean
			end
		end
	end
	mean = mean/(sum*(class[1]:size(2) - 1))
	local stddev = 0
	-- put dataset means to 0 and computes standard deviation
	for k = 1, 5 do
		for j = 1, class[k]:size(1) do
			for i = 2, class[k]:size(2) do -- we avoid the fist column,where the patch id is stored.
				class[k][j][i] = class[k][j][i] - mean
				stddev = stddev + class[k][j][i]*class[k][j][i]
			end
		end
	end
	stddev = math.sqrt(stddev/(sum*(class[1]:size(2) - 1)))
	--put standard deviation to 1
	for k = 1, 5 do
		for j = 1, class[k]:size(1) do
			for i = 2, class[k]:size(2) do -- we avoid the fist column,where the patch id is stored.
				class[k][j][i] = class[k][j][i]/stddev
			end
		end
	end
	for k = 1, 5 do
		torch.save('Brazil/class' .. tostring(k) .. '.data', class[k], ascii)
	end
end

if opt.u then
	print '==> charging genral dataset'
	--this one does not have parcel id.
	data = csvigo.load{path = '../Données et R/general_dataset.txt', separator = ' ', header = false, mode = 'raw'}
	print '==> processing general dataset'
	dataset = torch.Tensor(data)
	length = dataset:size(2)
	--data normalization
	local means = 0
	for k = 1, dataset:size(1) do
		for j = 1, length do
			means = means + dataset[k][j]
		end
	end
	means = means/(dataset:size(1)*length)
	-- put dataset means to 0 and computes standard deviation
	local stddev = 0
	for k = 1, dataset:size(1) do
		for i = 1, length do
			dataset[k][i] = dataset[k][i] - means
			stddev = stddev + dataset[k][i]*dataset[k][i]
		end
		
	end

	stddev = math.sqrt(stddev/(dataset:size(1)*length))
	--puts dataset standard deviation to 1
	for k = 1, dataset:size(1) do
		for i = 1, length do
			dataset[k][i] = dataset[k][i]/stddev
		end
	end

	--shuffle the dataset
dataset = dataset:index(1, torch.randperm(dataset:size(1)):long())
torch.save('Brazil/dataset.data', dataset, ascii)
end