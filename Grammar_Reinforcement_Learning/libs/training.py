import torch
from tqdm import tqdm


kl_loss = torch.nn.KLDivLoss(reduction = "batchmean")

def train(loader,rl_gram,optimiser,device,path,nb_epoch = 10000,stop = 150):
    rl_gram.train()
    best_loss = 1000000
    early_stop = 0
    for ep in tqdm(range(nb_epoch)):
        if early_stop> stop:
            break
        L1 = 0
        L2 = 0
        nb = 0
        for data in loader:
            optimiser.zero_grad()
            data = data.to(device)
            memory = rl_gram.encoder(data.var.T)
            pre = rl_gram.decoder(memory,data.state.T,src_mask = data.mask)[0,:,:]
            prob = torch.softmax(rl_gram.policy_mlp(pre),1)
            value = rl_gram.value_mlp(pre).squeeze(1)
            lss1 = kl_loss(torch.log(prob),data.prob)
            lss2 = torch.square(value-data.value).mean()
            lss = lss1+lss2
            lss.backward()
            optimiser.step()
            L1+=lss1.item()
            L2 += lss2.item()
            nb+=1
        L1 = L1/nb
        L2 = L2/nb
        if L1+L2< best_loss:
            best_loss = L1+L2
            early_stop = 0
            torch.save(rl_gram.state_dict(), path)
        else:
            early_stop +=1
        
        
        message = "Epoch {:04d}  policy loss:{:6.6f} value loss:{:6.6f} best loss:{:6.1f}"
        print(message.format(ep,L1,L2,best_loss))
    
        


    

        
        
    
