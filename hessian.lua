require 'nn'
require 'image'

-- Two ways to compute the Hessian, described in section 5.4.4 of Bishop's book.

function hessian(model, criterion, inputs, labels)
   w,dL_dw = model:getParameters()
   local n = w:size(1)
   local H = torch.Tensor(n,n)
   local eps = 1e-6
   for i = 1,n do
      if (i % 5) == 0 then
	 print(i .. ' / ' .. n)
      end
      -- function to perform gradient eval
      local feval = function(w)
         dL_dw:zero()
         local outputs = model:forward(inputs)
         local L = criterion:forward(outputs,labels)
         local dL_do = criterion:backward(outputs,labels)
         model:backward(inputs,dL_do)
         return dL_dw
      end
      w[i] = w[i] + eps
      H[i]:copy(feval(w))
      w[i] = w[i] - 2*eps
      H[i]:add(-1,feval(w))
      H[i]:div(2*eps)
      w[i] = w[i] + eps
   end
   return H
end


function hessian_finite_diff(model, criterion, inputs, labels)
   local w,dL_dw = model:getParameters()
   local n = w:size(1)
   local H = torch.Tensor(n,n)
   local eps = 1e-6
   local feval = function(w)
      dL_dw:zero()
      local outputs = model:forward(inputs)
      local L = criterion:forward(outputs,labels)
      return L
   end
   ei = torch.zeros(n)
   ej = torch.zeros(n)
   for i = 1,n do 
      print(i)
      for j = 1,n do
         w[i] = w[i] + eps
         w[j] = w[j] + eps
         -- 
         local L1 = feval()
         w[j] = w[j] - 2*eps
         local L2 = feval()
         w[i] = w[i] - 2*eps
         local L4 = feval()
         w[j] = w[j] + 2*eps
         local L3 = feval()
         w[j] = w[j] - eps
         w[i] = w[i] + eps
         H[i][j] = (L1-L2-L3+L4)/(4*eps*eps)
      end
   end
   return H
end

function test()
   local nsamples = 10000
   local dim = 100
   local nhidden = 25
   local nclasses = 10
   model = nn.Sequential()
   model:add(nn.Linear(dim,nhidden))
   model:add(nn.Threshold())
   model:add(nn.Linear(nhidden,nclasses))
   model:add(nn.LogSoftMax())
   criterion = nn.ClassNLLCriterion()
   inputs = torch.randn(nsamples,dim)
   labels = torch.Tensor(nsamples)
   for i = 1,nsamples do labels[i] =  math.random(1,nclasses) end
   local timer = torch.Timer()
   timer:reset()
   H1 = hessian(model,criterion,inputs,labels)
   print('First method takes' .. timer:time().real .. ' s')
   timer:reset()
   if false then
   H2 = hessian_finite_diff(model,criterion,inputs,labels)
   print('Second method takes' .. timer:time().real .. ' s')
   local err = torch.max(torch.abs(H1-H2))
   print('max error: ' .. err)
   return H1,H2
   end
end

--H1,H2 = test()


