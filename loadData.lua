-- script to load MNIST
-- pick subsample > 1 to subsample images
-- pick twoclass = true to restrict the dataset to 2 classes (digits 4 and 9)

require 'image'
require 'utils'

function loadData(dataset,split,subsample,twoclass) 
   local subsample = subsample or nil
   local f
   print(dataset)
   if dataset == 'cifar' then
      cifar = torch.load('cifar_10_norm.t7')
      if split == 'train' then 
	 f = cifar.train
      elseif split == 'test' then
	 f = cifar.test
      end
   elseif dataset == 'mnist' then
      if split == 'train' then
	 f = torch.load('mnist-torch7/train_28x28.th7nn')
      elseif split == 'test' then
	 f = torch.load('mnist-torch7/test_28x28.th7nn')
      else
	 error('set should be train or test')
      end
   end
   local labels = f.labels
   local data = f.data
   print(data:size())
   if subsample then
      print('subsampling')
      data2 = torch.Tensor(data:size(1),data:size(2),subsample,subsample)
      for i = 1,data:size(1) do 
	 data2[i]:copy(image.scale(data[i],subsample,subsample))
      end
      data = data2
   end
   data = data:resize(data:size(1),data:size(2)*data:size(3)*data:size(4))

   if twoclass then
      indx = torch.find(torch.eq(labels,5) + torch.eq(labels,10))
      data2 = torch.Tensor(indx:size(1),data:size(2))
      labels2 = torch.Tensor(indx:size(1))
      for i = 1,indx:size(1) do
	 data2[i]:copy(data[indx[i]])
	 if labels[indx[i]] == 5 then
	    labels2[i] = 1
	 else 
	    labels2[i] = 2
	 end
      end
      data = data2
      labels = labels2
   end
   return data:double(),labels:double()
end


-- show some images (note, we're assuming the data is [nSamples x n*n])
function showImages(data,nrows)
   local nSamples = data:size(1)
   local N = math.sqrt(data:size(2))
   imgs = data:resize(nSamples,N,N)
   image.display{image=imgs,nrows = nrows}
end
