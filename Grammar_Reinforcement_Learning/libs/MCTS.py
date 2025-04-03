import torch
import numpy as np
import psutil as ps


class Node(object):
    def __init__(self,parent,var,position,state,policy,value,leaf,cost,root = False,bad_leaf = False):
        super().__init__()
        self.root = root
        self.leaf = leaf
        if parent == None:
            self.parent = None
        else:
            self.parent = [parent]
        self.position = position
        self.state = state
        self.children = {}
        self.n = torch.tensor(0.)
        self.R = torch.tensor(0.)
        self.value = value
        self.policy = policy
        self.var = var
        self.cost = cost
        self.bad_leaf = bad_leaf

class Grammar_search(object):
    def __init__(self,grammar,nb_word,max_length = 30,nb_iter_mcts = 1000,value_confidence = 0.,search_param = 2.0**.5):
        super().__init__()
        self.grammar = grammar
        
        self.end = [grammar.dict_wtv[v] for v in grammar.variable]
    
        self.begin =  [torch.zeros((1,1),dtype = torch.int32)]*nb_word
        self.begin_word = str(sorted(self.grammar.vec_to_word(self.begin)))
        
        index = -torch.ones(len(grammar.dict_vtw))*float('inf')
        self.nb_iter_mcts = nb_iter_mcts
        self.value_confidence = value_confidence
        self.nb_word = nb_word
        self.max_length = max_length
        self.explore = {}
        self.explore_terminal = {}
        self.search_param = search_param
        
        for v in grammar.variable:
            tmp = index.clone()
            tmp2 = index.clone()
            tmp[torch.tensor(grammar.index_variable_rule[grammar.dict_wtv[v]])] = 0
            tmp2[torch.tensor(grammar.index_variable_terminal_rule[grammar.dict_wtv[v]])] = 0
            self.explore[grammar.dict_wtv[v]] = torch.softmax(tmp,0)
            self.explore_terminal[grammar.dict_wtv[v]] = torch.softmax(tmp2,0)
        self.nb_iter_mcts = nb_iter_mcts
        self.tree = {}
        
    def init_tree(self,rl_gram):
        policy,value = rl_gram(self.begin[0],torch.cat(self.begin),1)
        position,leaf = self.position_in_word(self.begin)
        self.tree.clear()
        self.tree = {self.begin_word:Node(None,self.begin[0],position,self.begin,policy,value,False,0,root = True)}
    
    def is_word(self,w):
        ret = torch.where(self.cond(w))
        return ret[0].shape[0]<1
    
    def cond(self,x):
        ret = False
        for val in self.end:
            ret += x == val
        return ret
    
    def position_in_word(self,state):
        for i,st in enumerate(state):
            ret = torch.where(self.cond(st.T))
            if ret[0].shape[0] > 0:
                ret = (ret[1][0:1],ret[0][0:1]+i)

                return ret,True
        return ret,False
    

        
    
    
    def MCTS(self,root,rl_gram,loader,ntask,nb_test = 10000,memory_min_free = .25,without_policy = False):
        for i in range(nb_test):
            if i> 5 and ps.virtual_memory().available/ ps.virtual_memory().total < memory_min_free:
                break
            leaf = self.select(root,rl_gram,without_policy = without_policy)
            sim_res = self.rollout(leaf,loader,ntask)
            self.backprop(leaf,sim_res)
        prob = root.policy*0
        tau = self.iprob(root.n)
        sum_prob = 0
        for child in root.children.values():
            sum_prob += self.tree[child].n**tau
        for act in root.children.keys():
            prob[act] = self.tree[root.children[act]].n**tau/(sum_prob)
        return self.best_child(root,without_policy=without_policy),prob
    

    def select(self,node,rl_gram,without_policy):
        while self.is_expand(node):
            node = self.tree[node.children[self.best_child(node,without_policy=without_policy)]]

        return self.expand(node,rl_gram,without_policy=without_policy)
    
    def is_expand(self,node):
        if node.position[1].shape[0]<1:
            return False
        if node.state[node.position[1]].shape[0]< self.max_length:
            return len(node.children) == len(self.grammar.index_variable_rule[node.var.item()])
        return len(node.children) == len(self.grammar.index_variable_terminal_rule[node.var.item()])
   
    def expand(self,node,rl_gram,without_policy):
        if node.leaf:
            return node
        explore = self.explore[node.var.item()].clone()
        if node.state[node.position[1]].shape[0]> self.max_length:
            explore = self.explore_terminal[node.var.item()].clone()
        for act in node.children.keys():
            explore[act] = 0
        explore = explore/explore.sum()
        action = torch.multinomial(explore,1)
        action_cost = 0
        if action.item() in self.grammar.action_cost:
            action_cost = self.grammar.action_cost[action.item()]
        rule = self.grammar.word_to_vec(self.grammar.dict_vtw[action.item()],len(self.grammar.dict_vtw[action.item()]))
        state = [st.clone() for st in node.state]
        state[node.position[1]] = torch.cat([state[node.position[1]][:node.position[0],:],rule,state[node.position[1]][node.position[0]+1:,:]])
        parent = str(sorted(self.grammar.vec_to_word(node.state)))
        expand_key = str(sorted(self.grammar.vec_to_word(state)))
        if expand_key in self.tree:
            self.tree[expand_key].parent.append(parent)
            node.children[action.item()] = expand_key
            return self.select(node,rl_gram,without_policy = without_policy)
        else:
            position,leaf = self.position_in_word(state)
            if leaf:
                var = state[position[1].item()][position[0]]
                policy,value = rl_gram(var,torch.cat(state),state[position[1].item()].shape[0])
            else:
                var = node.var
                policy ,value = rl_gram(var,torch.cat(state),state[node.position[1].item()].shape[0])
            bad_leaf =False
            if node.position[1].item()>0 and self.is_word(state[node.position[1].item()]):
                for st in state[:node.position[1].item()]:
                    if st.shape[0] ==  state[node.position[1].item()].shape[0] and torch.abs(st-state[node.position[1].item()]).sum()==0:
                        leaf = False
                        bad_leaf = True
                        break
                
            
            
            
            new = Node(parent,var,position,state,policy,value.detach(),not leaf,action_cost,bad_leaf = bad_leaf)
            if bad_leaf:
                new.R = torch.tensor(-1.)
            self.tree[expand_key] = new
            node.children[action.item()] = expand_key
            return new
    
    def rollout(self,leaf,loader,ntask):
        if leaf.bad_leaf:
            return -100.
        tr = not leaf.leaf
        if tr:
            var = leaf.var.clone()
        state = [st.clone() for st in leaf.state]
        leaf_key = str(sorted(self.grammar.vec_to_word(state)))
        if leaf.leaf and self.tree[leaf_key].n >0:
            return self.tree[leaf_key].R/self.tree[leaf_key].n
        position = leaf.position
        while tr:
            explore = self.explore[var.item()]
            if state[position[1].item()].shape[0]> self.max_length:
                explore = self.explore_terminal[var.item()]
            action = torch.multinomial(explore,1)
            rule = self.grammar.word_to_vec(self.grammar.dict_vtw[action.item()],len(self.grammar.dict_vtw[action.item()]))
            state[position[1]] = torch.cat([state[position[1]][:position[0],:],rule,state[position[1]][position[0]+1:,:]])
            if position[1].item()>0 and self.is_word(state[position[1].item()]):
                for st in state[:position[1].item()]:
                    if st.shape[0] ==  state[position[1].item()].shape[0] and torch.abs(st-state[position[1].item()]).sum()==0:
                        return -100.
            position,tr = self.position_in_word(state)
            if tr:
                var = state[position[1]][position[0]]
        return self.result(state,loader,ntask)
    
    def result(self,state,loader,ntask):
        word = self.grammar.vec_to_word(state)
        L = 0
        nb = 0
        alpha = None
        for data in loader:
            data= data
            out = torch.zeros(data.A.shape[0],len(state),data.A.shape[1],data.A.shape[2])
            test = (data.y[:,ntask,:,:]>0)*1.
            for i,w in enumerate(word):
                out[:,i,:,:] = self.grammar.calculatrice(w,data.A,data.I,data.J)
                test = test * (out[:,i,:,:]>0)
            test = torch.where(test==1)
            if test[0].shape[0] < out.shape[1]:
                t_ret = 0.
            else:
                M = torch.zeros((out.shape[1],out.shape[1]))
                for i in range(test[0].shape[0]-out.shape[1]):
                    M = out[test[0][i:i+out.shape[1]],:,test[1][i:i+out.shape[1]],test[2][i:i+out.shape[1]]]

                    if torch.abs(torch.linalg.det(M))> 1e-3:
                        alpha = torch.linalg.solve(M,data.y[test[0][i:i+out.shape[1]],ntask,test[1][i:i+out.shape[1]],test[2][i:i+out.shape[1]]])
                        break
                if alpha == None or torch.isnan(alpha.sum()):
                    t_ret = -100.
                else:
                    L += (torch.abs((alpha.unsqueeze(0).unsqueeze(2).unsqueeze(3)*out).sum(1)-data.y[:,ntask,:,:])/
                         (data.n_node.unsqueeze(1).unsqueeze(2))/(data.n_node.unsqueeze(1).unsqueeze(2))).sum((1,2)).mean()
                    nb += 1
        if nb>0:
            t_ret = 100*(torch.exp(-L/nb*6))
                
        return t_ret
    
    def backprop(self,node,result,gamma = .99):
        if node.root:
            return
        node.n += 1
        node.R += result-node.cost
        
        for key in node.parent:
            self.backprop(self.tree[key],gamma*(result-node.cost))
    
    def iprob(self,n,N = 15000000 ):
        if n>N:
            return np.log(N)/np.log(n)
        else:
            return 1.
    
    def best_child(self,node,without_policy):
        sum_n =  0 
        sum_prob = 0
        tree_conf = 1-self.value_confidence
        tau = self.iprob(node.n)
        for child in node.children.values():
            sum_n += self.tree[child].n
            
        
        if without_policy:
            choices_weights = [tree_conf*self.tree[node.children[k]].R/max(self.tree[node.children[k]].n,1)
                               +self.value_confidence*self.tree[node.children[k]].value
                               + self.search_param*np.sqrt(sum_n)/(self.tree[node.children[k]].n+1)
                               for k in list(node.children.keys())]
        else:
            choices_weights = [tree_conf*self.tree[node.children[k]].R/max(self.tree[node.children[k]].n,1)
                           +self.value_confidence*self.tree[node.children[k]].value
                           + self.search_param*node.policy[k]*np.sqrt(sum_n)/(self.tree[node.children[k]].n+1)
                           for k in list(node.children.keys())]
        ret = torch.argmax(torch.tensor(choices_weights))
        action = list(node.children.keys())[ret]
        return action
            
    def suppr_sub_tree(self,node):
        if len(self.tree[node].parent)>0:
            return
        else:
            if len(self.tree[node].children)>0:
                l_children = list(self.tree[node].children.values())
                for child in l_children:

                    del(self.tree[child].parent[self.tree[child].parent.index(node)])
                    self.suppr_sub_tree(child)
            del self.tree[node]
            
        
            
        
            
        


    

