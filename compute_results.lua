-- compute the eigenvalues of the hessian for all solutions computed for a given seed.

require 'nn'
dofile('loadData.lua')
dofile('hessian.lua')
dofile('accuracy.lua')

cmd = torch.CmdLine()
cmd:option('-dataset','cifar')
cmd:option('-seed',1)
cmd:option('-nlayers',1)
cmd:option('-nhidden',25)
cmd:option('-nexper',10)
cmd:option('-hessian',false)
cmd:option('-params',true)
opt = cmd:parse(arg or {})
print(opt)
torch.setnumthreads(1)
path = '/scratch/mbh305/criticalpoints/results/' .. opt.dataset ..  '/nlayers_' .. opt.nlayers .. '_nhidden_' .. opt.nhidden

if opt.dataset == 'cifar' then
   trdata,trlabels = loadData(opt.dataset,'train')
   tedata,telabels = loadData(opt.dataset,'test')
else
   trdata,trlabels = loadData(opt.dataset,'train',10)
   tedata,telabels = loadData(opt.dataset,'test',10)
end

criterion = nn.ClassNLLCriterion()
x = torch.load(path .. '/seed_' .. opt.seed .. '_epoch_100.th')
r = {}
for i = 1,opt.nexper do
   print('computing results for experiment ' .. i)
   r[i] = {}
   local model = x[i].model
   r[i].train_acc = accuracy(model:forward(trdata),trlabels)
   r[i].test_acc = accuracy(model:forward(tedata),telabels)
   r[i].train_loss = x[i].trainLoss
   r[i].test_loss = x[i].testLoss
   if opt.hessian then
      local H = hessian(model,criterion,trdata,trlabels)
      r[i].eig = torch.symeig(H)
   end
   if opt.params then
      r[i].params = model:getParameters()
   end
end

fname = path .. '/final_results_seed_' .. opt.seed 
if not opt.hessian then 
   fname = fname .. '_nohessian'
end
if opt.params then
   fname = fname .. '_params'
end

fname = fname .. '.th'
torch.save(fname,r)


   
   