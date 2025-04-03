import re
import torch

"""
A class that contains the dictionnaries of token corresponding to the input rules.

"""
class grammar(object):
    def __init__(self,rules,action_cost={},padding = 'P'):
        super().__init__()
        
        rules = rules.replace(" ","")
        l_rules = [st.strip() for st in rules.split('\n')if st.strip() !='']
        variables = [st[0] for st in l_rules]
        rule = [st[1:].replace("'",'').lstrip('->').split('|') for st in l_rules]
        terminals = set([st.replace("'",'') for st in re.findall('\'.\'',rules)])
        terminals_rules = []
        for v,st in zip(variables,rule):
            r = [s for s in st if v not in s]
            terminals_rules.append(r)
        non_rules_terminals = terminals
        for st in terminals_rules:
            non_rules_terminals = non_rules_terminals-set(st)
        
        
        dict_wtv = {}
        dict_vtw = {}
        i = 0
        for j,var in enumerate(variables):
            dict_wtv[var] = i
            dict_vtw[i] = var
            i+=1
            for rul in rule[j]:
                dict_wtv[rul] = i
                dict_vtw[i] = rul
                i+=1
        for term in non_rules_terminals:
            dict_wtv[term] = i
            dict_vtw[i] = term
            i+=1
        dict_wtv[padding] = i
        dict_vtw[i] = padding

        self.padding= padding        
        self.dict_wtv = dict_wtv
        
        self.dict_vtw = dict_vtw
        self.action_cost = {}
        for rul in action_cost.keys():
            if rul in self.dict_wtv:
                self.action_cost[self.dict_wtv[rul]] = action_cost[rul]
        self.index_variable_rule = {}
        self.index_variable_terminal_rule = {}
        for var,rul,term in zip(variables,rule,terminals_rules):
            self.index_variable_rule[self.dict_wtv[var]] = [self.dict_wtv[r] for r in rul]
            self.index_variable_terminal_rule[self.dict_wtv[var]] = [self.dict_wtv[r] for r in term]
        self.variable = variables
        self.terminals_rules = terminals_rules
        self.non_rules_terminals = non_rules_terminals

        
    def vec_to_word(self,vec):
        ret = []
        for j in vec:
            tmp = []
            for i in range(j.shape[0]):
                test = self.dict_vtw[int(j[i,0])]
                if test != self.padding:
                    tmp.append( self.dict_vtw[int(j[i,0])])
                
            ret.append("".join(tmp))
        return ret
    
    def word_to_vec(self,word,size):
        ret = torch.zeros((size,1))+self.dict_wtv[self.padding]
        for i, carac in enumerate(word):
            ret[i] = self.dict_wtv[carac]
        return ret.type(torch.int)
    
    def vec_to_vec(self,vec,size):
        ret = torch.zeros((size-vec.shape[0]),vec.shape[1])
        for i in range(size-vec.shape[0]):
            for j in range(vec.shape[1]):
                ret[i,j] = self.dict_wtv[self.padding]
        return torch.cat([vec,ret])
    
    def calculatrice(self,word,A,I,J):
        r = eval(word,{"A":A,"I":I,"J":J})
        return r

