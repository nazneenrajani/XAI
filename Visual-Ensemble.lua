require 'torch'
require 'nn'
require 'lfs'
require 'image'
require 'loadcaffe'
utils = require 'misc.utils'

files = {}
for file in io.lines('t') do
   i=1
   ttmp={}
   for word in string.gmatch(file, '([^\t]+)') do
        ttmp[i] = word
        i = i+1
   end
print(ttmp[1])
--local lstm = torch.load('forced-mcb-hiecoatt_gcam_output/2033350')--.. ttmp[1])
--local lstm_thresh = utils.threshold(lstm, 0.2)

local hiecoatt = torch.load('../HieCoAtt_gradcam/hie-sys2-mcb-hiecoatt_gcam_output/' .. ttmp[1])
local hiecoatt_thresh = utils.threshold(hiecoatt, 0.15)

local resnet = torch.load('../HieCoAtt_gradcam/res-sys2-mcb-hiecoatt_gcam_output/' .. ttmp[1])
local res_thresh = utils.threshold(resnet, 0.2)

--image.save('output-sys2-hm/h-' .. ttmp[1] ..'.png', image.toDisplayTensor(utils.to_heatmap(hiecoatt)))
--image.save('output-sys2-hm/l-' .. ttmp[1] ..'.png', image.toDisplayTensor(utils.to_heatmap(lstm)))
--image.save('output-sys2-lh/r-' .. ttmp[1] ..'.png', image.toDisplayTensor(utils.to_heatmap(resnet)))

--local sum = torch.add(hiecoatt,resnet)
--sum = torch.add(sum,lstm)
--local avg = torch.div(sum,3)

--image.save('output-sys2-hm/hmfl-'.. ttmp[1] .. '.png', image.toDisplayTensor(utils.to_heatmap(avg)))

sum = torch.add(hiecoatt_thresh,res_thresh)
--sum = torch.add(sum,lstm_thresh)
avg = torch.div(sum,2)
--torch.save('ensemble-gcam/e-w-'.. ttmp[1], avg)

--hiecoatt = torch.csub(hiecoatt,lstm)
--hiecoatt[hiecoatt:lt(0)] = 0
--resnet = torch.csub(resnet,lstm)
--resnet[resnet:lt(0)] = 0

--sum = torch.add(hiecoatt,resnet)
--avg = torch.div(sum,2)
--image.save('output-sys2-lh/lh-r-'.. ttmp[1] .. '.png', image.toDisplayTensor(utils.to_heatmap(avg)))
--torch.save('ensemble-gcam/e-'.. ttmp[1], avg)

--hiecoatt_thresh = torch.csub(hiecoatt_thresh,lstm_thresh)
--res_thresh = torch.csub(res_thresh,lstm_thresh)
--res_thresh[res_thresh:lt(0)] = 0
--hiecoatt_thresh[hiecoatt_thresh:lt(0)] = 0

--sum = torch.add(hiecoatt_thresh,res_thresh)
--avg = torch.div(sum,2)
torch.save('ensemble-gcam/hm-'.. ttmp[1], avg)
--image.save('output-sys2-hm/hmt-l-'.. ttmp[1] .. '.png', image.toDisplayTensor(utils.to_heatmap(avg)))

image.save('output/hm.png', image.toDisplayTensor(utils.to_heatmap(avg)))
--image.save('output/ltr.png', image.toDisplayTensor(utils.to_heatmap(lstm)))
--image.save('output/rt.png', image.toDisplayTensor(utils.to_heatmap(resnet)))
--local avg = torch.div(sum2,2)
--local avg_thresh1 = utils.threshold(avg)

--local avg_sum = torch.div(sum,2)
--local avg_thresh = utils.threshold(avg_sum)

--local hm =  utils.to_heatmap(avg_thresh)
--local hm1 =  utils.to_heatmap(avg_sum)
--local hm2 =  utils.to_heatmap(avg_thresh1)
--image.save('output/t-a-t.png', image.toDisplayTensor(hm))
--image.save('output/t-a.png', image.toDisplayTensor(hm1))
--image.save('output/a-t.png', image.toDisplayTensor(hm2))
end
