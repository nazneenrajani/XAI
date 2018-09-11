local utils = {}

-- Preprocess the image before passing it to a Caffe model.
function utils.preprocess(path, width, height)
  local width = width or 224
  local height = height or 224

  -- load image
  local orig_image = image.load(path)

  -- handle greyscale and rgba images
  if orig_image:size(1) == 1 then
    orig_image = orig_image:repeatTensor(3, 1, 1)
  elseif orig_image:size(1) == 4 then
    orig_image = orig_image[{{1,3},{},{}}]
  end

  -- get the dimensions of the original image
  local im_height = orig_image:size(2)
  local im_width = orig_image:size(3)

  -- scale and subtract mean
  local img = image.scale(orig_image, width, height):double()
  local mean_pixel = torch.DoubleTensor({103.939, 116.779, 123.68})
  img = img:index(1, torch.LongTensor{3, 2, 1}):mul(255.0)
  mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
  img:add(-1, mean_pixel)
  return img, im_height, im_width
end

-- Replace ReLUs with DeconvReLUs
function utils.deconv(m)
  require 'misc.DeconvReLU'
  local name = torch.typename(m)
  if name == 'nn.ReLU' or name == 'cudnn.ReLU' then
    return nn.DeconvReLU()
  else
    return m
  end
end

-- Replace ReLUs with GuidedBackpropReLUs
function utils.guidedbackprop(m)
  require 'misc.GuidedBackpropReLU'
  local name = torch.typename(m)
  if name == 'nn.ReLU' or name == 'cudnn.ReLU' then
    return nn.GuidedBackpropReLU()
  else
    return m
  end
end

-- Get layer id from name
function utils.cnn_layer_id(cnn, layer_name)
  for i = 1, #cnn.modules do
    local layer = cnn:get(i)
    local name = layer.name
    if name == layer_name then
      return i
    end
  end
  return -1
end

-- Synthesize gradInput tensor
function utils.create_grad_input(module, label)
  local doutput = module.output:clone():view(-1)
  doutput:fill(0)
  doutput[label] = 1
  return doutput
end

-- Creates gradInput for neuraltalk2 Language Model
function utils.create_grad_input_lm(input, labels)
  local output = torch.zeros(input:size()):fill(0)
  for t =1,labels:size(1) do
    if labels[t][1]~=0 then
      output[t+1][1][labels[t][1]] = 1
    end
  end
  return output
end

-- Generate Grad-CAM
function utils.grad_cam(cnn, layer_name, doutput)
  -- Split model into two
  local model1, model2 = nn.Sequential(), nn.Sequential()
  if tonumber(layer_name) == nil then

   for i = 1, #cnn.modules do
      model1:add(cnn:get(i))
      layer_id = i
      if cnn:get(i).name == layer_name then
        break
      end
    end
  else

    layer_id = tonumber(layer_name)
    for i = 1, #cnn.modules do
      model1:add(cnn:get(i))
      if i == layer_id then
        break
      end
    end
  end

  for i = layer_id+1, #cnn.modules do
    model2:add(cnn:get(i))
  end

  -- Get activations and gradients
  model2:zeroGradParameters()
  model2:backward(model1.output, doutput)
  
  -- Get the activations from model1 and and gradients from model2
  local activations = model1.output:squeeze()
  local gradients = model2.gradInput:squeeze()

  -- Global average pool gradients
  local weights = torch.sum(gradients:view(activations:size(1), -1), 2)

  -- Summing and rectifying weighted activations across depth
  local map = torch.sum(torch.cmul(activations, weights:view(activations:size(1), 1, 1):expandAs(activations)), 1)
  map = map:cmul(torch.gt(map,0):typeAs(map))

  return map
end

function utils.table_invert(t)
  local s = {}
  for k,v in pairs(t) do
    s[v] = k
  end
  return s
end

function utils.sent_to_label(vocab, sent, seq_length)
  local inv_vocab = utils.table_invert(vocab)
  local labels = torch.zeros(seq_length,1)
  local i = 0
  for word in sent:gmatch'%w+' do
    -- we replace out of vocabulary words with UNK
    if inv_vocab[word] == nil then
        word = 'UNK'
    end
    local ix_word = inv_vocab[word]
    i = i+1
    labels[{{i},{1}}] = ix_word
  end
  return labels
end

function utils.to_heatmap(map)
map = image.toDisplayTensor(map)
local cmap = torch.Tensor(3, map:size(2), map:size(3)):fill(1)
  for i = 1, map:size(2) do
    for j = 1, map:size(3) do
      local value = map[1][i][j]
      if value <= 0.25 then
        cmap[1][i][j] = 0
        cmap[2][i][j] = 4*value
      elseif value <= 0.5 then
        cmap[1][i][j] = 0
        cmap[3][i][j] = 2 - 4*value
      elseif value <= 0.75 then
        cmap[1][i][j] = 4*value - 2
        cmap[3][i][j] = 0
      else
        cmap[2][i][j] = 4 - 4*value
        cmap[3][i][j] = 0
      end
    end
  end
  return cmap
end


--[[ Below this added by nazneen ]]--

function utils.to_uncover_heatmap(map)
map = image.toDisplayTensor(map)
local cmap = torch.Tensor(3, map:size(2), map:size(3)):fill(1)
  for i = 1, map:size(2) do
    for j = 1, map:size(3) do
      local value = map[1][i][j]
	if value <= 0.4 then
        cmap[1][i][j] = 0
        cmap[2][i][j] = 0
        cmap[3][i][j] = 0
      elseif value <= 0.75 and value > 0.4 then
        cmap[1][i][j] = 4*value - 2
        cmap[3][i][j] = 0	
      elseif value >0.75 then
        cmap[2][i][j] = 4 - 4*value
        cmap[3][i][j] = 0
	end
    end
  end
  return cmap
end

function utils.uncover(map)
  map = image.toDisplayTensor(map)
  local cmap1 = torch.Tensor(1, map:size(2), map:size(3)):fill(0)
  local cmap2 = torch.Tensor(1, map:size(2), map:size(3)):fill(0)
  local cmap3 = torch.Tensor(1, map:size(2), map:size(3)):fill(0)
  local count = 0  
  local nz = torch.nonzero(map)
   for i = 1, map:size(2) do
    for j = 1, map:size(3) do
      if map[1][i][j] > 0 then
         count = count+1
      end
    end
  end
  local nz = torch.Tensor(1,count)
  count = 0
  for i = 1, map:size(2) do
    for j = 1, map:size(3) do
      if map[1][i][j] > 0 then
         count = count+1
         nz[1][count] = map[1][i][j]
      end
    end
  end
  s,id = torch.sort(nz)
  local med = torch.median(nz)
  local t1 = math.ceil(count/3)
  local t2 = t1*2
  local n1 = s[1][t1]
  local n2 = s[1][t2]
  for i = 1, map:size(2) do
    for j = 1, map:size(3) do
      if map[1][i][j] > 0 then
         cmap3[1][i][j] = map[1][i][j]
	 if map[1][i][j] > n1 then
	  cmap2[1][i][j] = map[1][i][j]
         if map[1][i][j] > n2 then
         cmap1[1][i][j] = map[1][i][j]
      end
	end
	end
    end
  end
  return cmap1,cmap2,cmap3
end

function utils.uncover_entire_image(map)
  map = image.toDisplayTensor(map)
  local cmap1 = torch.Tensor(1, map:size(2), map:size(3)):fill(0)
  local cmap2 = torch.Tensor(1, map:size(2), map:size(3)):fill(0)
  local cmap3 = torch.Tensor(1, map:size(2), map:size(3)):fill(0)
  local count = 0
  local nz = torch.nonzero(map)
   for i = 1, map:size(2) do
    for j = 1, map:size(3) do
      if map[1][i][j] > 0 then
         count = count+1
      end
    end
  end
  local nz = torch.Tensor(1,count)
  count = 0
  for i = 1, map:size(2) do
    for j = 1, map:size(3) do
      if map[1][i][j] > 0 then
         count = count+1
         nz[1][count] = map[1][i][j]
      end
    end
  end
  local nz1 =0 
  local nz2=0 
local nz3=0
  print(count)
  s,id = torch.sort(nz)
  local med = torch.median(nz)
  local t1 = math.ceil(count/3)
  local zcount = (224*224/4)
  t1 = math.ceil(224*224/4)
  if t1>count then
  	nz1 = t1 - count
  end
  local t2 = t1*2
  if t2>count then
        nz2 = t2 - count
  end
  local t3 = t1*3
  if t3>count then
        nz3 = t3 - count
  end
  print(nz1)
  print(nz2)
  print(nz3)
  local n1 = s[1][t1]
  local n2 = s[1][count]

  for i = 1, map:size(2) do
    for j = 1, map:size(3) do
         if map[1][i][j] > n1 then
           cmap1[1][i][j] = map[1][i][j]
  	   cmap2[1][i][j] = map[1][i][j]
	   cmap3[1][i][j] = map[1][i][j]
        end
	if map[1][i][j] > 0 then
	   cmap2[1][i][j] = map[1][i][j]
           cmap3[1][i][j] = map[1][i][j]
	end
    end
  end
 for i = 1, nz1 do
       x = torch.random(1,224)
         y = torch.random(1,224)
         cmap3[1][x][y] = 1.0
         cmap2[1][x][y] = 1.0
         cmap1[1][x][y] = 1.0
 	
  end
  for i = 1, nz2 do
       x = torch.random(1,224)
         y = torch.random(1,224)
         cmap3[1][x][y] = 1.0
          cmap2[1][x][y] = 1.0
  end
 
  for i = 1, zcount do
       x = torch.random(1,224)
         y = torch.random(1,224)
        cmap3[1][x][y] = 1.0
 end
  return cmap1,cmap2,cmap3
end

function utils.threshold(map,t)
  map = image.toDisplayTensor(map)
  local cmap = torch.Tensor(1, map:size(2), map:size(3)):fill(1)
  for i = 1, map:size(2) do
    for j = 1, map:size(3) do
      local value = map[1][i][j]
      if value <= t then
        cmap[1][i][j] = 0
      else
        cmap[1][i][j] = (value - t)/(1-t)
        --cmap[3][i][j] = 0
      end
    end
  end
  return cmap
end
return utils
