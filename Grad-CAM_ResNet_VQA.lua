require 'nn'
require 'torch'
require 'optim'
require 'misc.DataLoader'
require 'misc.word_level'
require 'misc.phrase_level'
require 'misc.ques_level'
require 'misc.recursive_atten'
require 'misc.cnnModel'
require 'misc.optim_updates'
utils = require 'misc.utils'
require 'xlua'
require 'image'


opt = {}

opt.vqa_model = '../VQA/HieCoAttenVQA/image_model/model_alternating_train-val_residual.t7'--'../gradcam-googlenet-resnet/fb.resnet.torch/resnet-152.t7'
opt.cnn_proto = '../VQA/ResNet-152-deploy.prototxt'
opt.cnn_model = '../VQA/ResNet-152-deploy.caffemodel'
opt.json_file = '../VQA/VQA_LSTM_CNN/vqa_data_prepro.json'--vqa_data_prepro_all.json'
opt.backend = 'cudnn'
opt.gpuid = 1
if opt.gpuid >= 0 then
  require 'cutorch'
  require 'cunn'
  if opt.backend == 'cudnn' then 
  require 'cudnn' 
  end
  --cutorch.setDevice(opt.gpuid+1) -- note +1 because lua is 1-indexed
end

local loaded_checkpoint = torch.load(opt.vqa_model)
local lmOpt = loaded_checkpoint.lmOpt

lmOpt.hidden_size = 512
lmOpt.feature_type = 'Residual'
lmOpt.atten_type = 'Alternating'
cnnOpt = {}
cnnOpt.cnn_proto = opt.cnn_proto
cnnOpt.cnn_model = opt.cnn_model
cnnOpt.backend = opt.backend
cnnOpt.input_size_image = 512
cnnOpt.output_size = 512
cnnOpt.h = 14
cnnOpt.w = 14
cnnOpt.layer_num = 37

-- load the vocabulary and answers.

local json_file = utils.read_json(opt.json_file)
ix_to_word = json_file.ix_to_word
ix_to_ans = json_file.ix_to_ans
ans_to_ix = utils.table_invert(ix_to_ans)
word_to_ix = {}
for ix, word in pairs(ix_to_word) do
    word_to_ix[word]=ix
end

-- load the model
protos = {}
protos.word = nn.word_level(lmOpt)
protos.phrase = nn.phrase_level(lmOpt)
protos.ques = nn.ques_level(lmOpt)

protos.atten = nn.recursive_atten()
protos.crit = nn.CrossEntropyCriterion()
--protos.cnn = nn.cnnModel(cnnOpt)
protos.cnn = torch.load('../gradcam-googlenet-resnet/fb.resnet.torch/resnet-200.t7')
for i = 14,12,-1 do
    protos.cnn:remove(i)
end
cnn_gb = protos.cnn:clone()
cnn_gb:replace(utils.guidedbackprop)
cnn_gb:evaluate()
protos.cnn:evaluate()

if opt.gpuid >= 0 then
  for k,v in pairs(protos) do v:cuda() end
end

cparams, grad_cparams = protos.cnn:getParameters()
wparams, grad_wparams = protos.word:getParameters()
pparams, grad_pparams = protos.phrase:getParameters()
qparams, grad_qparams = protos.ques:getParameters()
aparams, grad_aparams = protos.atten:getParameters()

print('Load the weight...')
wparams:copy(loaded_checkpoint.wparams)
pparams:copy(loaded_checkpoint.pparams)
qparams:copy(loaded_checkpoint.qparams)
aparams:copy(loaded_checkpoint.aparams)

print('total number of parameters in cnn_model: ', cparams:nElement())
assert(cparams:nElement() == grad_cparams:nElement())

print('total number of parameters in word_level: ', wparams:nElement())
assert(wparams:nElement() == grad_wparams:nElement())

print('total number of parameters in phrase_level: ', pparams:nElement())
assert(pparams:nElement() == grad_pparams:nElement())

print('total number of parameters in ques_level: ', qparams:nElement())
assert(qparams:nElement() == grad_qparams:nElement())
protos.ques:shareClones()

print('total number of parameters in recursive_attention: ', aparams:nElement())
assert(aparams:nElement() == grad_aparams:nElement())

-- specify the image and the question.
--COCO_val2014_000000094538 -3
--COCO_val2014_000000284743 -2, COCO_val2014_000000374010 -1
--local img_path = 'input_images/COCO_val2014_000000094538.jpg'
--local question = 'what is that ?'
--local answer = nil
-- load the image
files = {}
opt.dir = '/scratch/cluster/nrajani/VQA/test2015/'
for file in io.lines('tmp') do
   i=1
   ttmp={}
   for word in string.gmatch(file, '([^\t]+)') do
        ttmp[i] = word
        i = i+1
   end
--for i = 1,#tmp do
   local question = ttmp[3]
   local img = image.load('/scratch/cluster/nrajani/VQA/test2015/COCO_test2015_' .. string.format("%012d",tonumber(ttmp[2])) .. '.jpg')
   local answer = ttmp[4]
--local img = image.load(img_path)
if img:size(1) ==4 then print("changing dim")
    img = img[{{1,3},{},{}}] end

-- scale the image
img = image.scale(img,448,448)
--itorch.image(img)

--image.save('/scratch/cluster/nrajani/VQA/HieCoAtt_gradcam/orig3.png', img)
img = img:view(1,img:size(1),img:size(2),img:size(3))
-- parse and encode the question (in a simple way).
local ques_encode = torch.IntTensor(26):zero()

local count = 1
for word in string.gmatch(question, "%S+") do
    ques_encode[count] = word_to_ix[word] or word_to_ix['UNK']
    count = count + 1
end
ques_encode = ques_encode:view(1,ques_encode:size(1))
-- doing the prediction

protos.word:evaluate()
protos.phrase:evaluate()
protos.ques:evaluate()
protos.atten:evaluate()
protos.cnn:evaluate()

local image_raw = utils.prepro_residual(img:squeeze(), false)

image_raw = image_raw:cuda()
ques_encode = ques_encode:cuda()
local image_feats_raw = protos.cnn:forward(image_raw)
local img_feats_gb = cnn_gb:forward(image_raw)
image_feat = image_feats_raw:permute(1,3,4,2):contiguous()
image_feat = image_feats_raw:view(1,196,2048)
local ques_len = torch.Tensor(1,1):cuda()
ques_len[1] = count-1

local word_feat, img_feat, w_ques, w_img, mask = unpack(protos.word:forward({ques_encode, image_feat}))

local conv_feat, p_ques, p_img = unpack(protos.phrase:forward({word_feat, ques_len, img_feat, mask}))

local q_ques, q_img = unpack(protos.ques:forward({conv_feat, ques_len, img_feat, mask}))

local feature_ensemble = {w_ques, w_img, p_ques, p_img, q_ques, q_img}
local out_feat = protos.atten:forward(feature_ensemble)

local tmp,pred=torch.max(out_feat,2)
local ans = ix_to_ans[tostring(pred[1][1])]


-- forward the language model criterion
--local loss = protos.crit:forward(out_feat, data.answer)
-----------------------------------------------------------------------------
-- Backward pass
-----------------------------------------------------------------------------
-- backprop criterion


--local dlogprobs = protos.crit:backward(out_feat, data.answer)
print("ans_to_ix[answer]",ans_to_ix[answer])
print("answer",answer)
if answer ~= nil and ans_to_ix[answer] ~= nil then 
    answer_idx = ans_to_ix[answer] 
    print("using given answer")
else 
    answer = ans 
    answer_idx = ans_to_ix[answer] 
    print("using pred answer")
end
local doutput = out_feat:view(-1):fill(0)
print("answer_idx:", answer_idx)
doutput[answer_idx] = 1
doutput = doutput:view(1,1000)
--local doutput = utils.create_grad_input(protos.atten.modules[#protos.atten.modules], answer_idx)
  
local d_w_ques, d_w_img, d_p_ques, d_p_img, d_q_ques, d_q_img = unpack(protos.atten:backward(feature_ensemble, doutput))

local d_ques_feat, d_ques_img = unpack(protos.ques:backward({conv_feat, ques_len, img_feat}, {d_q_ques, d_q_img}))
    
--local d_ques1 = protos.bl1:backward({ques_feat_0, data.ques_len}, d_ques2)
local d_conv_feat, d_conv_img = unpack(protos.phrase:backward({word_feat, ques_len, img_feat}, {d_ques_feat, d_p_ques, d_p_img}))


local dcnn = protos.word:backward({ques_encode, image_feat}, {d_conv_feat, d_w_ques, d_w_img, d_conv_img, d_ques_img})

--print("dcnn:shape", dcnn:size())
dcnn = dcnn:view(1,14,14,2048)
local activations = image_feats_raw:squeeze()
--print("activations max and min: ", activations:max(), activations:min())

local gradients = dcnn:squeeze():permute(3,1,2):contiguous()
--print("gradients max and min: ", gradients:max(), gradients:min())

--gradients = gradients:cmul(torch.gt(gradients, 0):typeAs(gradients))
local weights = torch.sum(gradients:view(activations:size(1), -1), 2):squeeze()
--weights = weights:cmul(torch.gt(weights, 0):typeAs(weights))

map = torch.sum(torch.cmul(activations, weights:view(activations:size(1), 1, 1):expandAs(activations)), 1)
--print("map max and min: ",map:max(), map:min())
map = image.scale(map:float(), 224, 224)

--itorch.image(utils.to_heatmap(map))

map = map:cmul(torch.gt(map, 0):typeAs(map))
gcam = image.scale(map:float(), 224, 224)

--gcampos = gcam
--image.save('output/'..answer..'_gcam.png', image.toDisplayTensor(gcam))
--gcam = utils.grad_cam(protos.cnn,4,dcnn)
---gcam = image.scale(gcam:float(), 448, 448)
local hm = utils.to_heatmap(gcam)
torch.save('vis-tmp/r-'..ttmp[1],gcam)
image.save('vis-tmp/r-' ..ttmp[1]..'.png', image.toDisplayTensor(hm))
end
--print('The answer is: ' .. ans)
--dcnn = dcnn:permute(1,4,2,3)


