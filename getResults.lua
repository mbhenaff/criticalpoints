require 'nn'

path = '/scratch/mbh305/CriticalPoints/Results/nlayers_1_nhidden_3'

trainLoss = torch.Tensor(5000)
testLoss = torch.Tensor(5000)
w0 = torch.Tensor(5000,41)
w = torch.Tensor(5000,41)

indx = 1
for s = 1,50 do
	local x = torch.load(path .. '/seed_' .. s .. '_epoch_1000.th')
	for exp = 1,100 do
		local m0 = x[exp].modelInit:getParameters()
		local m = x[exp].model:getParameters()
		w0[indx]:copy(m0)
		w[indx]:copy(m)
		trainLoss[indx]=x[exp].trainLoss
		testLoss[indx] = x[exp].testLoss
		indx = indx + 1
		print(indx)
	end
end
