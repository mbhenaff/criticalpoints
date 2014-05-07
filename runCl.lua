require 'optim'
require 'nn'
dofile('loadData.lua')


-- process command line options
cmd = torch.CmdLine()
cmd:option('-nhidden',3)
cmd:option('-nlayers',1)
cmd:option('-epochs',500)
cmd:option('-seed',1)
cmd:option('-nexper',1,'number of experiments')
opt = cmd:parse(arg or {})
print(opt)

torch.manualSeed(opt.seed)

--------------------------------------------------
-- LOAD DATASET
--------------------------------------------------

data = torch.load('mnist-torch7/10PC.th')
trlabels = data.labels
trdata = data.data
data = torch.load('mnist-torch7/10PC_test.th')
telabels = data.labels
tedata = data.data

nSamples = trdata:size(1)
nInputs = trdata:size(2)

------------------------------------------------------------------
-- DEFINE MODEL
------------------------------------------------------------------
nClasses = 2
nHidden = opt.nhidden

model = nn.Sequential()
model:add(nn.Reshape(nInputs))
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
	learningRate = 0.01,
	weightDecay = 0,
	learningRateDecay = 1/20000
}

-------------------------------------------
-- TRAIN
-------------------------------------------
epochs = 100
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
		local Ltrain = criterion:forward(outputs,trlabels)
		outputs = model:forward(tedata)
		local Ltest = criterion:forward(outputs,telabels)
		print('Epoch ' .. i .. ' | Train Loss = ' .. Ltrain .. ' | Test Loss = ' .. Ltest)	
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
end	

-- save results
filename = '/scratch/mbh305/CriticalPoints/Results/nlayers_' .. opt.nlayers .. '_nhidden_' .. opt.nhidden .. '/seed_' .. opt.seed .. '_epoch_' .. epochs .. '.th'
os.execute('mkdir -p ' .. sys.dirname(filename))
torch.save(filename, results)
















