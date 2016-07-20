----------------------------------------------------------------------

require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'lfs'	  -- allows changing the current directory
----------------------------------------------------------------------

if not opt then
	print 'MAIN: processing options'
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Options:')
	cmd:option('-directory', 'Brazil', 'selects the directory to use')
	cmd:option('-subdirectory', "", 'the subdirectory where the first convolution has been trained')
	cmd:option('-save', "", 'wether or not the neural network should be saved.')
	cmd:option('-nfilter', 10, 'the number of filters in the 1st convolution')
	cmd:option('-nfilter2', -1, 'the number of the 2nd convlution')
	cmd:option('-k_2', 5, 'kernel of the 2nd convolution')
	cmd:option('-p_2', 2, 'second pooling size')
	cmd:text()
end

opt = cmd:parse( arg or {})
-- saving the writing of the number of filters
if opt.nfilter2 == -1 then 
	nfilter2 = tonumber(opt.save)
end

here = lfs.currentdir()
if not lfs.chdir(here..'/'..opt.directory..'/'..opt.subdirectory..'/'..opt.save) then--change the working directory to the dataset directory
	error('fail in directory change')
end
if not pcall(function () torch.load('model2.data', ascii) end) then --there is no model constructed here.
	old_model = torch.load('../model1.data',ascii)
	first_layer = old_model:get(1)
	second_layer = nn.Sequential()
	conv2 = nn.TemporalConvolution(opt.nfilter, nfilter2, opt.k_2)
	second_layer:add(conv2)
	second_layer:add(nn.Sigmoid())
	second_layer:add(nn.TemporalMaxPooling(opt.p_2))

	model = nn.Sequential()
	model:add(first_layer)
	model:add(second_layer)
	pass = 9 -- the last pass completed
	while pass > 0 do
		-- we go back in the pass numbers and pick the first completed one.
		if pcall(function () torch.load('ae2_'..pass..'.data',ascii) end) then
			ae = torch.load('ae2_'..pass..'.data',ascii)
			pass = 0
			params2, grad_params2 = ae:get(2):getParameters() --selects the encoder part before extracting
			cparams2, cgrad_params2 = conv2:getParameters()
			print(#params2)
			print(#grad_params2)
			print(#cparams2)
			print(#cgrad_params2)
			for k = 1, params2:size(1) do
				cparams2[k] = params2[k]
				cgrad_params2[k] = grad_params2[k]
			end

			if opt.save ~= "" then
				torch.save('model2.data', model, ascii) -- saving the model
			end
		else
			pass = pass - 1
		end
	end
end