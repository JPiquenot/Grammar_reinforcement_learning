import torch
import libs.grammartools as gt

from libs.utils import GraphCountnodeDataset,Grammardataset
from torch_geometric.loader import DataLoader
from libs.grammar_model import rl_grammar
from tqdm import tqdm
import libs.MCTS as mcts
import libs.actor as act
import libs.training as tr
import gc
import tracemalloc
import sys
import argparse
import os



# parser = argparse.ArgumentParser()



# parser.add_argument('pre_learning',default = 'True')

# args = parser.parse_args()

# pre_learning = args.pre_learning
pre_learning = True

# tracemalloc.start()


rules = """
        E -> '('E'*'M')' | '('N'@'E')' | '('E'@'N')' | 'A' | 'J'
        N -> '('N'*'M')' | '('N'*'N')' | 'I'
        M -> '('M'@'M')' | '('E'@'E')'
        """

action_cost = {'(E@E)':1.3,
                '(M@M)':1.3,
                '(N@E)':1.3,
                '(E@N)':1.3,
                '(E*M)':1.2,
                '(N*M)':1.1,
                '(N*N)':1.1}
# rules = """
#         M -> '('M'@'M')' | '('M'*'M')'  | 'A' | 'I' | 'J' 
#         """

G = gt.grammar(rules,action_cost)




graph_dataset = GraphCountnodeDataset(root="dataset/subgraphcount/")

split = int(len(graph_dataset))
rest_split = len(graph_dataset) - split



dt, rest_dt = torch.utils.data.random_split(graph_dataset,[split,rest_split])


graph_batch_size = 1024

lr_rl = 5*1e-5
lr_w = 1e-3

graph_loader = DataLoader(graph_dataset[dt.indices], batch_size=graph_batch_size, shuffle=True)


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


max_depth = 45
nb_word = 4
batch_size = 2048



# for episode in range(100):
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
rl_gram = rl_grammar(G,device,nb_word = nb_word,d_model = 128,nhead = 4,d_hid = 128, 
             num_layers = 5, dropout = 0.,max_depth=max_depth).to(device)

if  pre_learning :
    #rl_gram.load_state_dict(torch.load("save/grammartest/grammar.dat",map_location=torch.device('cpu')))
    rl_gram.load_state_dict(torch.load("save/grammartest/grammar.dat"))

    # tree.value_confidence = .05

seq = []
pa = ''
pattern = '(?<=score:)( )+\d+.\d'
for f in tqdm(os.listdir(pa+'save/')):
    if "agent" in f:
        maxim = 0
        for fil in os.listdir(pa+'save/'+f):
            # if 'result' in fil:
            #     path = 'save/'+f+'/'+fil
            #     with open(path,'r') as red:
            #         maxim = float(re.search(pattern,red.read()).group(0).strip(' '))
            #     
            # if maxim>30 and 'save' in fil:
            #     path = 'save/'+f+'/'+fil
            if 'save' in fil:
                path = pa+'save/'+f+'/'+fil
                seq += torch.load(path)
            




    
print('saving dataset') 


print(device)    
    
torch.save(seq,'dataset/gram/raw/savetest.dat')
dataset = Grammardataset('dataset/gram/',G)
# print(dataset.data)

loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)




optimiser = torch.optim.Adam(rl_gram.parameters(), lr=lr_rl,maximize = False)

tr.train(loader, rl_gram, optimiser, device, 'save/grammartest/grammar.dat')







