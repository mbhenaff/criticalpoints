require 'torch'
require 'gnuplot'

dataset = 'mnist'
nlayers = 1
nhidden = {2,3,5,10,25,50}
r = {}

for i = 1,4 do
   path = '/scratch/mbh305/criticalpoints/results/' .. dataset ..  '/nlayers_' .. nlayers .. '_nhidden_' .. nhidden[i]
   r[i] = torch.load(path .. '/final_results_all.th')
end


