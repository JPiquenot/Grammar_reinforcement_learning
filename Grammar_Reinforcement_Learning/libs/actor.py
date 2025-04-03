import torch



class actor(object):
    def __init__(self,ntask):
        super().__init__()
        self.ntask = ntask


    def episode(self,root,mcts_search,rl_grammar,loader,without_policy):
        sequence = [(str(sorted(mcts_search.grammar.vec_to_word(root.state))),0,1)]
        
        while not root.leaf:
            root.root = True
            # print(len(mcts_search.tree))
            action,prob = mcts_search.MCTS(root,rl_grammar,loader,self.ntask,nb_test = mcts_search.nb_iter_mcts,without_policy = without_policy)
            # print(len(mcts_search.tree))
            node = mcts_search.tree[root.children[action]]
            print(root.value,root.R/max(root.n,1),prob,mcts_search.grammar.vec_to_word(root.state),root.policy)
            
            
            sequence.append((root.children[action],action,prob))

            # l_children = list(root.children.keys())
            # for child in l_children:
            #     if child != action:
            #         mcts_search.tree[root.children[child]].parent = []
            #         mcts_search.suppr_sub_tree(root.children[child])
            # del l_children
            # root.root=False
            # root.children = {action:root.children[action]}
            root = node
        
        return sequence,sequence[-1][0],root.R/root.n

    def create_sequence(self,sequence,mcts_search):
        seq = []
        for no in sequence:
            node = mcts_search.tree[no[0]]
            #root = mcts_search.tree[sequence[i][0]]
            #node = mcts_search.tree[sequence[i+1][0]]
            st = torch.cat(node.state)
            var = node.var
            prob = node.policy * 0
            tau = mcts_search.iprob(node.n)
            sum_prob = 0
            for child in node.children.values():
                sum_prob += mcts_search.tree[child].n**tau
            for act in node.children.keys():
                prob[act] = mcts_search.tree[node.children[act]].n**tau/sum_prob
            #prob = sequence[i+1][2]
            #action = sequence[i+1][1]
            dic = {'word' : mcts_search.grammar.vec_to_word(node.state) ,
                   #'action': action,
                   'prob' : prob,
                   'value': node.R/max(1,node.n),
                   #'value_root': node.R/max(1,node.n),
                   'var': var,
                   'state': st,
                   'len': st.shape[0]
                   }
            seq.append(dic)
        return seq
    
    def acting(self,mcts_search,rl_grammar,loader,nb_episode = 100,path ='dataset/gram',without_policy = False):
        rl_grammar.eval()
        sequence = []
        max_reward = -100
        best_word = ['']
        
        for ep in range(nb_episode):
            mcts_search.init_tree(rl_grammar)
            root = mcts_search.tree[mcts_search.begin_word]
            seq,word,value = self.episode(root,mcts_search,rl_grammar,loader,without_policy)
            seq = self.create_sequence(seq,mcts_search)
            sequence = sequence + seq
            torch.save(sequence,path + 'raw/savetest.dat')
            if value>= max_reward:
                best_word = word
                max_reward = value
            message = "Episode {:04d}  score:{:6.6f} best score:{:6.1f} word:"
            print(message.format(ep,value,max_reward) + str(word))
            print(best_word)

                
            


