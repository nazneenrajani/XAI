require 'torch'
require 'image'
utils = require 'misc.utils'


files = {}
opt.dir = '/scratch/cluster/nrajani/grad-cam'

for file in io.lines('testdev_sys1_qid') do
   i=1
   ttmp={}
   for word in string.gmatch(file, '([^\t]+)') do
        ttmp[i] = word
        i = i+1
   end
print(ttmp[1])

local lstm_thresh = utils.threshold(lstm,0.2)
local hiecoatt = torch.load('../HieCoAtt_gradcam/hiecoatt_gcam_output-sys3/' .. ttmp[1])
local hiecoatt_thresh = utils.threshold(hiecoatt,0.15)
local resnet = torch.load('../HieCoAtt_gradcam/res_sys1_gcam_output/')
local res_thresh = utils.threshold(lstm,0.2)

-local intersection = torch.Tensor(1,224,224):fill(0)
local union = torch.Tensor(1,224,224):fill(0)

for x = 1, lstm:size(2) do
    for y = 1, lstm:size(3) do
    	union[{1,x,y}] = math.max(lstm_thresh[{1,x,y}], res_thresh[{1,x,y}], hiecoatt_thresh[{1,x,y}])
    end
end

for x = 1, lstm:size(2) do
    for y = 1, lstm:size(3) do
	if (lstm_thresh[{1,x,y}] > torch.mean(lstm_thresh)  and res_thresh[{1,x,y}] > torch.mean(res_thresh)  and hiecoatt_thresh[{1,x,y}] > torch.mean(hiecoatt_thresh)) then
		intersection[{1,x,y}] = math.max(lstm_thresh[{1,x,y}], res_thresh[{1,x,y}], hiecoatt_thresh[{1,x,y}])
	end
    end
end

--[[uncov1, uncov2,uncov3 = utils.uncover(lstm_thresh)--]]
uncov1, uncov2,uncov3 = utils.uncover_entire_image(lstm_thresh)
image.save('uncover-entireimage-sys3/c1-l-'.. ttmp[1] .. '.png', image.toDisplayTensor(uncov1))
image.save('uncover-entireimage-sys3/c2-l-'.. ttmp[1] .. '.png', image.toDisplayTensor(uncov2))
image.save('uncover-entireimage-sys3/c3-l-' .. ttmp[1] .. '.png', image.toDisplayTensor(uncov3))

uncov1, uncov2,uncov3 = utils.uncover_entire_image(hiecoatt_thresh)
image.save('uncover-entireimage-sys3/c1-h-'.. ttmp[1] .. '.png', image.toDisplayTensor(uncov1))
image.save('uncover-entireimage-sys3/c2-h-'.. ttmp[1] .. '.png', image.toDisplayTensor(uncov2))
image.save('uncover-entireimage-sys3/c3-h-'.. ttmp[1] .. '.png', image.toDisplayTensor(uncov3))

uncov1, uncov2,uncov3 = utils.uncover_entire_image(res_thresh)
image.save('c1-l-2359102.png', image.toDisplayTensor(uncov1))
image.save('c2-l-2359102.png', image.toDisplayTensor(uncov2))
image.save('c3-l-2359102.png', image.toDisplayTensor(uncov3))

uncov1, uncov2,uncov3 = utils.uncover_entire_image(ensemble)
image.save('c1-e-1881291.png', image.toDisplayTensor(uncov1))
image.save('c2-e-1881291.png', image.toDisplayTensor(uncov2))
image.save('c3-e-1881291.png', image.toDisplayTensor(uncov3))

uncov1, uncov2,uncov3 = utils.uncover(res_thresh)
image.save('uncover-sys1/c1-r-'.. ttmp[1] .. '.png', image.toDisplayTensor(uncov1))
image.save('uncover-sys1/c2-r-'.. ttmp[1] .. '.png', image.toDisplayTensor(uncov2))
image.save('uncover-sys1/c3-r-'.. ttmp[1] .. '.png', image.toDisplayTensor(uncov3))


uncov1, uncov2,uncov3 = utils.uncover(ensemble)
image.save('uncover-sys1/c1-e-p-'.. ttmp[1] .. '.png', image.toDisplayTensor(uncov1))
image.save('uncover-sys1/c2-e-p-'.. ttmp[1] .. '.png', image.toDisplayTensor(uncov2))
image.save('uncover-sys1/c3-e-p-'.. ttmp[1] .. '.png', image.toDisplayTensor(uncov3))

end
