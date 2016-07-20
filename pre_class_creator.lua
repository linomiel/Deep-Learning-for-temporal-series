require 'torch'   -- torch
require 'nn'      -- provides all sorts of trainable modules/layers
require 'lfs'	  -- allows changing the current directory

if not opt then
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Options:')
	cmd:option('-d', '/Brazil', 'selects the directory to use')
	cmd:text()
end

opt = cmd:parse( arg or {})

here = lfs.currentdir()
if not lfs.chdir(here..opt.d) then--change the working directory to the model directory
	error('fail in directory change')
end

file = io.open("classes.txt", "w")
for k = 1, 5 do
	class = torch.load('class'..tostring(k)..'.data', ascii)
	for i = 1, class:size(1) do
		local temp = class[1]:narrow(1,2,class:size(2)-1)
		file:write(k .. " ")
		for j = 1, temp:size(1) do
			file:write(tostring(temp[j]), " ")
		end
		file:write("\n")
	end
end
file:close()