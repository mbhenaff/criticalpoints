-- script to load MNIST
-- pick subsample > 1 to subsample images
-- pick twoclass = true to restrict the dataset to 2 classes (digits 4 and 9)

require 'image'
require 'utils'

function loadData(set,subsample,twoclass)
	subsample = subsample or 1
	if set == 'train' then
		mnist = torch.load('mnist-torch7/train_28x28.th7nn')
	elseif set == 'test' then
		mnist = torch.load('mnist-torch7/test_28x28.th7nn')
	else
		error('set should be train or test')
	end
	local labels = mnist.labels
	local data = mnist.data
	data = data:squeeze()
	if subsample > 1 then
		data2 = torch.Tensor(data:size(1),data:size(2)/subsample,data:size(2)/subsample)
		for i=1,data:size(2)/subsample do
			for j=1,data:size(2)/subsample do
				data2[{{},i,j}]:copy(data[{{},1+(i-1)*subsample,1+(j-1)*subsample}])
			end
		end
		data = data2
	end
	data = data:resize(data:size(1),data:size(2)*data:size(3))

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
	return data,labels
end


-- show some images (note, we're assuming the data is [nSamples x n*n])
function showImages(data,nrows)
	local nSamples = data:size(1)
	local N = math.sqrt(data:size(2))
	imgs = data:resize(nSamples,N,N)
   image.display{image=imgs,nrows = nrows}
end













