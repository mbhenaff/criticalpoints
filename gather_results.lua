require 'nn'
require 'gnuplot'
dofile('csv.lua')

cmd = torch.CmdLine()
cmd:option('-dataset','mnist')
cmd:option('-seed',1)
cmd:option('-nlayers',1)
cmd:option('-nhidden',25)
cmd:option('-nexper',10)
cmd:option('-hessian',true)
opt = cmd:parse(arg or {})
print(opt)

path = '/scratch/mbh305/criticalpoints/results/' .. opt.dataset ..  '/nlayers_' .. opt.nlayers .. '_nhidden_' .. opt.nhidden

function file_exists(name)
   local f=io.open(name,"r")
   if f~=nil then io.close(f) return true else return false end
end

eps = 0.001
nseeds = 100
nexper = 10

train_loss = torch.Tensor(nseeds*nexper)
test_loss = torch.Tensor(nseeds*nexper)
train_acc = torch.Tensor(nseeds*nexper)
test_acc = torch.Tensor(nseeds*nexper)
indx = torch.Tensor(nseeds*nexper)

--x = torch.load(path .. '/final_results_seed_' .. 1 .. '_nohessian_params.th')
--n = x[1].params:nElement()
--params = torch.Tensor(nseeds*nexper,n)

cntr = 1
missing = 0
for s = 1,nseeds do
   if opt.nexper == 1 then
      for e = 1,10 do 
	 local fname = path .. '/final_results_seed_' .. s .. '_experiment_' .. e .. '.th'
	 if file_exists(fname) then
	    x = torch.load(fname)
	    train_loss[cntr] = x[1].train_loss
	    test_loss[cntr] = x[1].test_loss
	    train_acc[cntr] = x[1].train_acc
	    test_acc[cntr] = x[1].test_acc
	    --params[cntr] = x[i].params
	    indx[cntr] = torch.sum(torch.lt(x[1].eig,-eps)) / x[1].eig:nElement()
	    cntr = cntr + 1
	 else
	    print('missing ' .. fname)
	    missing = missing + 1
	 end
      end
   else   
      --local fname = path .. '/final_results_seed_' .. s .. '_nohessian_params.th'
      local fname = path .. '/final_results_seed_' .. s .. '.th'
      if file_exists(fname) then
	 x = torch.load(fname)
	 for i = 1,nexper do
	    train_loss[cntr] = x[i].train_loss
	    test_loss[cntr] = x[i].test_loss
	    train_acc[cntr] = x[i].train_acc
	    test_acc[cntr] = x[i].test_acc
	    --params[cntr] = x[i].params
	    indx[cntr] = torch.sum(torch.lt(x[i].eig,-eps)) / x[i].eig:nElement()
	    cntr = cntr + 1
	 end
      else
	 print('missing ' .. fname)
	 missing = missing + 1
      end
   end
end

if missing > 0 then
   print('warning, missing ' .. missing .. ' results')
   train_loss = train_loss[{{1,cntr-1}}]
   test_loss = test_loss[{{1,cntr-1}}]
   train_acc = train_acc[{{1,cntr-1}}]
   test_acc = test_acc[{{1,cntr-1}}]
   indx = indx[{{1,cntr-1}}]
end

--print('mean for train_acc: ' .. torch.mean(train_acc))
--print('mean for test_acc : ' .. torch.mean(test_acc))
--print('std for train_acc : ' .. torch.std(train_acc))
--print('std for test_acc  : ' .. torch.std(test_acc))

torch.save(path .. '/final_results_all.th',{train_loss=train_loss,test_loss=test_loss,train_acc=train_acc,test_acc=test_acc})
writeCSV(path .. '/final_results_train_loss.csv',train_loss)
writeCSV(path .. '/final_results_test_loss.csv',test_loss)
writeCSV(path .. '/final_results_train_acc.csv',train_acc)
writeCSV(path .. '/final_results_test_acc.csv',test_acc)
writeCSV(path .. '/final_results_indx.csv',indx)
