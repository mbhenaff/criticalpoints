require 'torch'

function accuracy(outputs,labels)
   local nsamples = labels:size(1)
   local pred=torch.Tensor(nsamples)
   local correct = 0
   for i = 1,nsamples do 
      local m,indx = torch.sort(outputs[i],true)
      pred[i] = indx[1]
      if pred[i] == labels[i] then 
	 correct = correct + 1
      end
   end
   return correct / nsamples
end
