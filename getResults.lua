require 'nn'
dofile('loadData.lua')
dofile('hessian.lua')

cmd = torch.CmdLine()
cmd:option('-dataset','cifar')
cmd:option('-seed',1)
cmd:option('-nlayers',1)
cmd:option('-nhidden',25)
cmd:option('-nexper',10)
cmd:option('-hessian',false)
opt = cmd:parse(arg or {})
print(opt)

path = '/scratch/mbh305/criticalpoints/results/' .. opt.dataset ..  '/nlayers_' .. opt.nlayers .. '_nhidden_' .. opt.nhidden

trdata,trlabels = loadData(opt.dataset,'train',10)
tedata,telabels = loadData(opt.dataset,'test',10)
criterion = nn.ClassNLLCriterion()

trainLoss = torch.Tensor(1000)
testLoss = torch.Tensor(1000)
w0 = torch.Tensor(5000,41)
w = torch.Tensor(5000,41)
indx = 1
for s = 1,100 do
	x = torch.load(path .. '/seed_' .. s .. '_epoch_100.th')
	for exp = 1,10 do
		--local m0 = x[exp].modelInit:getParameters()
		--local m = x[exp].model:getParameters()
		--w0[indx]:copy(m0)
		--w[indx]:copy(m)
	   trainLoss[indx]=x[exp].trainLoss
	   testLoss[indx] = x[exp].testLoss
	   indx = indx + 1
	   print(indx)
	end
end

torch.save(path .. '/results.th',{train_loss=trainLoss,test_loss=testLoss})