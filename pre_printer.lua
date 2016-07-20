--[[--------------------------------------------
This File is for Brazilian data only. It precomputes the 
classes' data and writes them in treated_patches.txt for
python to load it with json.
This file operates on the raw data. For a file operating on the post CNN data, see printer.lua
----------------------------------------------]]
require 'torch'
require 'nn'
require 'lfs'	  -- allows changing the current directory

if not opt then
	cmd = torch.CmdLine()
	cmd:text()
	cmd:text('Options:')
	cmd:option('-d', '/Brazil', 'selects the directory to use')
end

opt = cmd:parse( arg or {})

here = lfs.currentdir()
if not lfs.chdir(here..opt.d) then--change the working directory to the model directory
	error('fail in directory change')
end


file = io.open("treated_patches.txt", "w")
target = io.open("target_patches.txt", "w")
--opening the classes list
file:write("[")
target:write("[")
--treating each class
for k = 1, 5 do
	current_class = torch.load("class"..k..".data", ascii)
	--file:write("[[") -- opening class and first patch
	file:write("[") --no more class opening
	eoc = current_class:size(1)
	for i = 1, eoc do
		local temp = current_class[i]:narrow(1,2,current_class:size(2) - 1)
		file:write("[")--opening pixel features
		for j = 1, temp:size(1) do
			file:write(tostring(temp[j]))
			if j < temp:size(1) then
				file:write(", ")
			end
		end
		file:write("]")--closing pixel features
		if i < eoc and current_class[i+1][1] ~= current_class[i][1] then -- thanks shortcut evaluation! intermédiaite patch change
			file:write("], [") -- next patch
			target:write(k..", ") --saving current patch
		elseif i < eoc then -- same patch, not final value
			file:write(", ")--too next pixel
		else --final patch end
			target:write(tostring(k))
		end
	end
	--file:write("]]")--closing last patch and class
	file:write("]")--no moe closing class
	if k < 5 then
		file:write(", ")
		target:write(", ")
	end
end
--closing the classes list
file:write("]")
target:write("]")
file:close()
target:close()