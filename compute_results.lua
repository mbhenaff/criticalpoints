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
cmd:option('-hessian',true)
cmd:option('-params',false)
cmd:option('-exper',-1)
opt = cmd:parse(arg or {})
print(opt)
torch.setnumthreads(1)

if opt.nexper == 1 and opt.exper == -1 then
   error('must specify experiment number')
end

path = '/scratch/mbh305/criticalpoints/results/' .. opt.dataset ..  '/nlayers_' .. opt.nlayers .. '_nhidden_' .. opt.nhidden

if opt.dataset == 'cifar' then
   trdata,trlabels = loadData(opt.dataset,'train')
   tedata,telabels = loadData(opt.dataset,'test')
elseif opt.dataset == 'cifar_10x10' then
   trdata,trlabels = loadData('cifar','train',10)
   tedata,telabels = loadData('cifar','test',10)
else
   trdata,trlabels = loadData(opt.dataset,'train',10)
   tedata,telabels = loadData(opt.dataset,'test',10)
end

criterion = nn.ClassNLLCriterion()
r = {}
x = torch.load(path .. '/seed_' .. opt.seed .. '_epoch_100.th')
if opt.nexper == 1 then
   r[1] = {}
   local model = x[opt.exper].model
   r[1].train_acc = accuracy(model:forward(trdata),trlabels)
   r[1].test_acc = accuracy(model:forward(tedata),telabels)
   r[1].train_loss = x[opt.exper].trainLoss
   r[1].test_loss = x[opt.exper].testLoss
   if opt.hessian then
      local H = hessian(model,criterion,trdata,trlabels)
      r[1].eig = torch.eig(H)
   end
   if opt.params then
      r[1].params = model:getParameters()
   end
else
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
	 r[i].eig = torch.eig(H)
      end
      if opt.params then
	 r[i].params = model:getParameters()
      end
      collectgarbage()
   end
end

if opt.nexper > 1 then
   fname = path .. '/final_results_seed_' .. opt.seed 
else
   fname = path .. '/final_results_seed_' .. opt.seed .. '_experiment_' .. opt.exper 
end
if not opt.hessian then 
   fname = fname .. '_nohessian'
end
if opt.params then
   fname = fname .. '_params'
end

fname = fname .. '.th'
torch.save(fname,r)


   
   