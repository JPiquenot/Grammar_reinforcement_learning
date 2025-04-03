import torch
import libs.transformers as tr





class rl_grammar(torch.nn.Module):
    def __init__(self,grammar,device,nb_word = 1,d_model = 128,nhead = 4,d_hid = 128, 
                 num_layers = 5, dropout = 0,max_depth = 30):
        super().__init__()
    
        
        self.encoder = tr.grammar_encoder(len(grammar.dict_vtw),d_model = d_model,nhead = nhead,d_hid = d_hid, 
                     num_layers = num_layers, dropout = dropout)
        
        self.decoder = tr.grammar_decoder(len(grammar.dict_vtw),d_model = d_model,nhead = nhead,d_hid = d_hid, 
                     num_layers = num_layers, dropout = dropout)
        
        # index_mlp = []
        # for v in grammar.variable:
        #     index_mlp.append(grammar.dict_wtv[v])
        
        # for v in grammar.non_rules_terminals:
        #     index_mlp.append(grammar.dict_wtv[v])
            
        # self.index_mlp =torch.tensor(index_mlp,dtype = torch.int64,device = device)
        
        self.value_mlp = torch.nn.Sequential(torch.nn.Linear(d_model,256),
                                             torch.nn.ReLU(),
                                              torch.nn.Linear(256,256),
                                              torch.nn.ReLU(),
                                             torch.nn.Linear(256,1))
        
        self.policy_mlp = torch.nn.Sequential(torch.nn.Linear(d_model,256),
                                             torch.nn.ReLU(),
                                              torch.nn.Linear(256,256),
                                              torch.nn.ReLU(),
                                             torch.nn.Linear(256,len(grammar.dict_vtw)))
            
        

        index = -torch.ones(len(grammar.dict_vtw),device=device)*float('inf')

        self.variable_token = {}
        self.variable_token_terminal = {}
        
        for v in grammar.variable:
            tmp = index.clone()
            tmp2 = index.clone()
            tmp[torch.tensor(grammar.index_variable_rule[grammar.dict_wtv[v]])] = 0
            tmp2[torch.tensor(grammar.index_variable_terminal_rule[grammar.dict_wtv[v]])] = 0
            self.variable_token[grammar.dict_wtv[v]] = tmp
            self.variable_token_terminal[grammar.dict_wtv[v]] = tmp2
        self.max_depth = max_depth
        self.padding = grammar.dict_wtv[grammar.padding]   
        
    def forward(self,state,bword,length):

        memory = self.encoder(state)
        val = self.decoder(memory,bword)
        value = self.value_mlp(val[0,:,:]).squeeze(0)
        policy = self.policy_mlp(val[0,:,:]).squeeze(0)
        var = state[0,0].item()

       

        if length> self.max_depth:
            policy = torch.softmax(policy + self.variable_token_terminal[var],0)
        else:
            policy = torch.softmax(policy + self.variable_token[var],0)
        return policy,value

