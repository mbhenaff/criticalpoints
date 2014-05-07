
--------------------------------------------
-- SET OPERATIONS ON TABLES
--------------------------------------------   

set={}

function set.new(t)
   local set={}
   for _,l in ipairs(t) do set[l]=true end
   return set
end

function set.union(a,b)
   local res=Set.new{}
   for k in pairs(a) do res[k]=true end
   for k in pairs(b) do res[k]=true end
   return res
end

function set.intersect(a,b)
   local res=Set.new{}
   for k in pairs(a) do
      res[k]=b[k]
   end
   return res
end

function set.difference(a,b)
   local res=Set.new{}
   for k in pairs(a) do
      if not b[k] then res[k]=true end
   end
return res
end

function set.table(s)
   local t={}
   for k in pairs(s) do
      t[#t+1]=k
   end
   return t
end

function any(x)
   return torch.sum(torch.ne(x,0))>0
end

-- convert the 1D tensor x to a table
function torch.Table(x)
   if x ~= nil then
      local t={}
      for i=1,(#x)[1] do
         table.insert(t,x[i])
      end
      return t
   else
      return {}
   end
end


---------------------------------------------------------
-- some MATLAB-like set operations on 1D tensors
---------------------------------------------------------

-- set difference of two tensors (result is sorted)  
function torch.setdiff(a,b)
   local res=set.table(Set.difference(Set.new(torch.Table(a)),Set.new(torch.Table(b))))
   table.sort(res)
   return torch.Tensor(res)
end

-- set union of two tensors
function torch.union(a,b)
   local res=set.table(Set.union(Set.new(torch.Table(a)),Set.new(torch.Table(b))))
   table.sort(res)
   return torch.Tensor(res)
end

-- set intersection of two tensors 
function torch.intersect(a,b)
   local res=set.table(Set.intersect(Set.new(torch.Table(a)),Set.new(torch.Table(b))))
   table.sort(res)
   return torch.Tensor(res)
end

-- returns the indices of non-zero elements of a 1D tensor 
function torch.find(x)
   if x:nDimension() > 1 then
		error('torch.find is only defined for 1D tensors')
	end
   local indx={}
   for i=1,(#x)[1] do
      if x[i]>0 then
         table.insert(indx,i)
      end
   end
   return torch.IntTensor(indx)
end



 

-- load data from a csv file and convert to tensor
function load_data(filename,row,col,delim)
   local row=row or 1
   local col=col or 1
   local delim=delim or ','
   require 'csv'
   local data = csv.load{path=filename,mode='raw',separator=delim}
   local n=#data
   local d=#data[1]
   local D=torch.Tensor(n,d)
   for i=row,n do
      for j=col,d do
         D[i][j]=data[i][j]
      end
   end
   return D
end


-- performance metrics
function accuracy(targets,predictions)
   correct=0
   incorrect=0
   predictions=torch.lt(predictions,0.5)
   correct=torch.sum(torch.eq(targets,predictions:type('torch.DoubleTensor')))
   return correct/targets:size()[1]
end

function auc(targets,pred)
   local neg=pred[torch.ne(targets,1)]
   local pos=pred[torch.eq(targets,1)]   
	if neg:nElement() == 0 or pos:nElement() == 0 then
		print('warning, there is only one class')
	end
   local C=0
   for i=1,(#pos)[1] do
      for j=1,(#neg)[1] do
         if neg[j]<pos[i] then
            C=C+1
         elseif neg[j]==pos[i] then
            C=C+0.5
         end
      end
   end
   local AUC=C/((#neg)[1]*(#pos)[1])
   return AUC
end

      
      
      












   
