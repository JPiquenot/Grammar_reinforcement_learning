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



parser = argparse.ArgumentParser()


parser.add_argument('agent_nb',type=int)
parser.add_argument('ntask',type=int)
parser.add_argument('without_policy',default = 'False')

args = parser.parse_args()
nb_agent = args.agent_nb
without_policy = args.without_policy == 'True'


# tracemalloc.start()
ntask = args.ntask


rules = """
        E -> '('E'*'M')' | '('N'@'E')' | '('E'@'N')' | 'A' | 'J'
        N -> '('N'*'M')' | '('N'*'N')' | 'I'
        M -> '('M'@'M')' | '('E'@'E')'
        """

action_cost = {'(E@E)':3.,
                '(M@M)':3.,
                '(N@E)':3.,
                '(E@N)':3.,
                '(E*M)':2.,
                '(N*M)':1.,
                '(N*N)':1.}
# rules = """
#         M -> '('M'@'M')' | '('M'*'M')'  | 'A' | 'I' | 'J' 
#         """

G = gt.grammar(rules,action_cost)


path = 'save/agent'+str(nb_agent)
save_res = path + '/results_agent' + str(nb_agent) + '.dat'
save_data = path + '/save_agent' + str(nb_agent) + '.dat'

if not os.path.exists(path):
    os.makedirs(path)

graph_dataset = GraphCountnodeDataset(root="dataset/subgraphcount/")

split = int(len(graph_dataset))
rest_split = len(graph_dataset) - split



dt, rest_dt = torch.utils.data.random_split(graph_dataset,[split,rest_split])


graph_batch_size = 1024

lr_rl = 2*1e-6
lr_w = 1e-3

graph_loader = DataLoader(graph_dataset[dt.indices], batch_size=graph_batch_size, shuffle=True)


nb_episode = 1
nb_iter_mcts = 10000
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


max_depth = 45
nb_word = 4
batch_size = 512
value_confidence = 0.
search_param = 10.

tree = mcts.Grammar_search(G,nb_word,nb_iter_mcts = nb_iter_mcts ,value_confidence = value_confidence ,search_param=search_param)
agent = act.actor(ntask)
# for episode in range(100):
device = torch.device('cpu')
rl_gram = rl_grammar(G,device,nb_word = nb_word,d_model = 128,nhead = 8,d_hid = 128, 
             num_layers = 5, dropout = 0.,max_depth = max_depth).to(device)

if not without_policy:
    rl_gram.load_state_dict(torch.load("save/grammartest/grammar_3path.dat",map_location=torch.device('cpu')))
    tree.value_confidence = .01

# print(tree.value_confidence)


rl_gram.eval()
with torch.no_grad():
    sequence = []
    max_reward = -100
    best_word = ['']
    
    tree.init_tree(rl_gram)
    root = tree.tree[tree.begin_word]
    seq,word,value = agent.episode(root,tree,rl_gram,graph_loader,without_policy)
    sequence = agent.create_sequence(seq,tree)
    torch.save(sequence,save_data)
    if value>= max_reward:
        best_word = word
        max_reward = value
    message = "Agent {:04d}  score:{:6.6f} best score:{:6.4f} word:"
    with open(save_res,'w') as res:
        res.write(message.format(nb_agent,value,max_reward) + str(word) + "\n")
    # agent.acting(tree,rl_gram,graph_loader,nb_episode=nb_episode,path = 'dataset/gram/',without_policy = without_policy)

    
    
    # # torch.save(seq,'dataset/gram/raw/savetest.dat')
    # dataset = Grammardataset('dataset/gram/',G)
    # # print(dataset.data)
    
    # loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # print(sys.getrefcount(device),sys.getrefcount(rl_grammar))
    # del rl_gram
    # rl_gram = rl_grammar(G,device,nb_word = nb_word,d_model = 128,nhead = 4,d_hid = 1024, 
    #              num_layers = 5, dropout = 0.).to(device)
    # # print(sys.getrefcount(device),sys.getrefcount(rl_grammar))
    # if episode >0:
    #     rl_gram.load_state_dict(torch.load("save/grammartest/grammar.dat",map_location=torch.device('cpu')))
    
    # optimiser = torch.optim.Adam(rl_gram.parameters(), lr=lr_rl,maximize = False)
    
    # tr.train(loader, rl_gram, optimiser, device, 'save/grammartest/grammar.dat')
    # del optimiser
    # del rl_gram






