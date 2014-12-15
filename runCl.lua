require 'optim'
require 'nn'
require 'nnx'
require 'optim'
require 'image'
dofile('loadData.lua')
dofile('accuracy.lua')

-- process command line options
cmd = torch.CmdLine()
cmd:option('-dataset','mnist')
cmd:option('-model','mlp')
cmd:option('-nhidden',25)
cmd:option('-nlayers',1)
cmd:option('-epochs',100)
cmd:option('-seed',1)
cmd:option('-nexper',10,'number of experiments')
cmd:option('-learningRate',0.01)
cmd:option('-weightDecay',0)
opt = cmd:parse(arg or {})
print(opt)

torch.manualSeed(opt.seed)
torch.setnumthreads(1)
--------------------------------------------------
-- LOAD DATASET
--------------------------------------------------
if opt.dataset == 'reuters' then
   trdata,trlabels = loadData('reuters','train')
   tedata,telabels = loadData('reuters','test')            
elseif opt.dataset == 'cifar_10x10' then 
   trdata,trlabels = loadData('cifar','train',10)
   tedata,telabels = loadData('cifar','test',10)      
elseif opt.dataset == 'cifar' then 
   trdata,trlabels = loadData('cifar','train')
   tedata,telabels = loadData('cifar','test')            
else
   trdata,trlabels = loadData('mnist','train',10)
   tedata,telabels = loadData('mnist','test',10)
end
nSamples = trdata:size(1)
nInputs = trdata:size(2)
------------------------------------------------------------------
-- DEFINE MODEL
------------------------------------------------------------------
if opt.dataset == 'reuters' then
   nClasses = 50
else
   nClasses = 10
end
nHidden = opt.nhidden

print(opt.model)

if opt.model == 'mlp' then
   model = nn.Sequential()
   --model:add(nn.Reshape(nInputs))
   model:add(nn.Linear(nInputs,nHidden))
   model:add(nn.Threshold())
   for i = 1,opt.nlayers-1 do
      model:add(nn.Linear(nHidden,nHidden))
      model:add(nn.Threshold())
   end
elseif opt.model == 'convnet' then
   model = nn.Sequential()
   model:add(nn.Reshape(3,32,32))
   -- stage 0 : RGB -> YUV -> normalize(Y)
   --model:add(nn.SpatialColorTransform('rgb2yuv'))
   do
      ynormer = nn.Sequential()
      ynormer:add(nn.Narrow(1,1,1))
      ynormer:add(nn.SpatialContrastiveNormalization(1, image.gaussian1D(7)))
      normer = nn.ConcatTable()
      normer:add(ynormer)
      normer:add(nn.Narrow(1,2,2))
   end
   --model:add(normer)
   --model:add(nn.JoinTable(1))
   -- stage 1 : mean+std normalization -> filter bank -> squashing -> max pooling
   --local table = torch.Tensor{ {1,1},{1,2},{1,3},{1,4},{1,5},{1,6},{1,7},{1,8},{2,9},{2,10},{3,11},{3,12} }
   model:add(nn.SpatialConvolution(3,opt.nhidden,5,5))
   --model:add(nn.SpatialConvolutionMap(table, 5, 5))
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   -- stage 2 : filter bank -> squashing -> max pooling
   model:add(nn.SpatialConvolution(opt.nhidden,opt.nhidden,5,5))
   --model:add(nn.SpatialConvolutionMap(nn.tables.random(12, 32, 4), 5, 5))
   model:add(nn.Threshold())
   model:add(nn.SpatialMaxPooling(2, 2, 2, 2))
   -- stage 3 : standard 2-layer neural network
   --model:add(nn.Reshape(opt.nhidden*14*14))
   --model:add(nn.Linear(opt.nhidden*14*14,10))
   model:add(nn.Reshape(opt.nhidden*5*5))
   --model:add(nn.Linear(opt.nhidden*5*5, opt.nhidden))
   --model:add(nn.Tanh())
   model:add(nn.Linear(opt.nhidden*5*5,10))
else
   error('unrecognized model: ' .. opt.model)
end

model:add(nn.Linear(nHidden,nClasses))

-- loss
model:add(nn.LogSoftMax())
criterion = nn.ClassNLLCriterion()

-----------------------------------------
-- OPTIMIZATION
-----------------------------------------
params = {
   learningRate = opt.learningRate,
   weightDecay = 0,
   learningRateDecay = 1/trdata:size(1),
   weightDecay = opt.weightDecay
}

-------------------------------------------
-- TRAIN
-------------------------------------------
epochs = opt.epochs
batchSize = 1

-- tensors to store minibatch
inputs = torch.Tensor(batchSize,nInputs)
targets = torch.Tensor(batchSize)

results = {}
timer = torch.Timer()
print(opt)

if true then

troutputs = torch.Tensor(trdata:size(1), nClasses)
teoutputs = torch.Tensor(tedata:size(1), nClasses)

-- loop over number of experiments
for exp = 1,opt.nexper do
   timer:reset()
   params.evalCounter = 0
   -- pick new initialization point
   model:reset()
   -- get weight and gradient vectors
   w,dL_dw = model:getParameters()
   results[exp] = {}
   results[exp].modelInit = model:clone()

   -- loop over epochs
   for i = 1,epochs do
      local shuffle = torch.randperm(nSamples)
      -- loop over minibatches
      for t = 1,(nSamples-batchSize),batchSize do
	 if (t % 1000) == 0 then
	    --print(t)
	 end
	 -- create minibatch
	 for i = 1,batchSize do
	    inputs[{i,{}}]:copy(trdata[shuffle[t+i]])
	    targets[i]=trlabels[shuffle[t+i]]
	 end	
	 -- create closure to evaluate loss and gradient
	 local feval = function(w_)
			  if w_ ~= w then
			     w:copy(w_)
			  end
			  -- reset gradients
			  dL_dw:zero()
			  -- fprop/bprop
			  local outputs = model:forward(inputs)
			  local L = 0
			  L = criterion:forward(outputs, targets)
			  local dL_do = criterion:backward(outputs,targets)
			  model:backward(inputs,dL_do)	
			  return L,dL_dw
		       end
	 -- optimize on current minibatch
	 optim.sgd(feval,w,params)
      end	
      for i = 1,trdata:size(1) do 
	 troutputs[i]:copy(model:forward(trdata[i]))
      end
      --local outputs = model:forward(trdata)
      local tracc = accuracy(troutputs,trlabels)
      local Ltrain = criterion:forward(troutputs,trlabels)
      for i = 1,tedata:size(1) do 
 	 teoutputs[i]:copy(model:forward(tedata[i]))
      end
      --outputs = model:forward(tedata)
      local Ltest = criterion:forward(teoutputs,telabels)
      local teacc = accuracy(teoutputs,telabels)
      print('Epoch ' .. i .. ' | Train Loss = ' .. Ltrain .. ' | Test Loss = ' .. Ltest .. ' | Train accuracy = ' .. tracc .. ' | Test accuracy = ' .. teacc)	
   end
   -- compute final loss over training and test sets
   for i = 1,trdata:size(1) do 
      troutputs[i]:copy(model:forward(trdata[i]))
   end
   local Ltrain = criterion:forward(troutputs,trlabels)

   for i = 1,tedata:size(1) do 
      teoutputs[i]:copy(model:forward(tedata[i]))
   end
   --outputs = model:forward(tedata)
   local Ltest = criterion:forward(teoutputs,telabels)
   -- record results
   results[exp].model = model:clone()
   results[exp].trainLoss = Ltrain
   results[exp].testLoss = Ltest
   print('\nExperiment took ' .. timer:time().real .. ' seconds\n')
   collectgarbage()
end	

-- save results
filename = '/scratch/mbh305/criticalpoints/results/' .. opt.dataset .. '/nlayers_' .. opt.nlayers .. '_nhidden_' .. opt.nhidden .. '/seed_' .. opt.seed .. '_epoch_' .. epochs .. '.th'
os.execute('mkdir -p ' .. sys.dirname(filename))
torch.save(filename, results)

end