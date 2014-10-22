require 'optim'
require 'nn'
dofile('loadData.lua')
dofile('accuracy.lua')

-- process command line options
cmd = torch.CmdLine()
cmd:option('-dataset','cifar')
cmd:option('-nhidden',25)
cmd:option('-nlayers',1)
cmd:option('-epochs',100)
cmd:option('-seed',1)
cmd:option('-nexper',1,'number of experiments')
cmd:option('-learningRate',0.01)
cmd:option('-weightDecay',0)
opt = cmd:parse(arg or {})
print(opt)

torch.manualSeed(opt.seed)
torch.setnumthreads(1)
--------------------------------------------------
-- LOAD DATASET
--------------------------------------------------
if false then 
   data = torch.load('mnist-torch7/10PC.th')
   trlabels = data.labels:double()
   trdata = data.data:double()
   data = torch.load('mnist-torch7/10PC_test.th')
   telabels = data.labels:double()
   tedata = data.data:double()
else
   if opt.dataset == 'cifar' then 
      trdata,trlabels = loadData('cifar','train')
      tedata,telabels = loadData('cifar','test')      
   else
      trdata,trlabels = loadData('mnist','train',10)
      tedata,telabels = loadData('mnist','test',10)
   end
end
nSamples = trdata:size(1)
nInputs = trdata:size(2)
print(trdata:size())
print(tedata:size())
------------------------------------------------------------------
-- DEFINE MODEL
------------------------------------------------------------------
nClasses = 10
nHidden = opt.nhidden

model = nn.Sequential()
--model:add(nn.Reshape(nInputs))
model:add(nn.Linear(nInputs,nHidden))
model:add(nn.Threshold())

for i = 1,opt.nlayers-1 do
   model:add(nn.Linear(nHidden,nHidden))
   model:add(nn.Threshold())
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
      local outputs = model:forward(trdata)
      local tracc = accuracy(outputs,trlabels)
      local Ltrain = criterion:forward(outputs,trlabels)
      outputs = model:forward(tedata)
      local Ltest = criterion:forward(outputs,telabels)
      local teacc = accuracy(outputs,telabels)
      print('Epoch ' .. i .. ' | Train Loss = ' .. Ltrain .. ' | Test Loss = ' .. Ltest .. ' | Train accuracy = ' .. tracc .. ' | Test accuracy = ' .. teacc)	
   end
   -- compute final loss over training and test sets
   local outputs = model:forward(trdata)
   local Ltrain = criterion:forward(outputs,trlabels)
   outputs = model:forward(tedata)
   local Ltest = criterion:forward(outputs,telabels)
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