-- compute principal components of MNIST
nPC = 10
dofile('loadData.lua')
data,labels = loadData('train')

nFeatures = data:size(2)
-- center
means = torch.mean(data,1)
data:add(-means:expandAs(data))
-- compute covariance and project on PCs
cov = data:t() * data
eval,evec = torch.symeig(cov,'V')
evec = evec[{{},{nFeatures-nPC+1,nFeatures}}]
proj = data * evec
-- see how much info is lost
recon = proj * evec:t()
showImages(recon[{{1,64},{}}])
torch.save('mnist-torch7/10PC.th',{data=proj,labels=labels,evec = evec})




