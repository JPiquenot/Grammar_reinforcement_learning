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
        # torch.cuda.empty_cache()
        nb = 0
        for data in loader:
            optimiser.zero_grad()
            data = data.to(device)
            memory = rl_gram.encoder(data.var.T)
            pre = rl_gram.decoder(memory,data.state.T,src_mask = data.mask)[0,:,:]
            prob = torch.softmax(rl_gram.policy_mlp(pre),1)
            # prob = prob[torch.arange(prob.shape[0]),data.action]
            value = rl_gram.value_mlp(pre).squeeze(1)
            # print(data.value_root.std())
            # print(value.shape,data.value.shape,data.value_root.shape,data.prob.shape)
            # ind = torch.where(data.value_root >98.)
            # print(data.value_root.mean())
            # if ind[0].shape[0]>0:
            #     print(value[ind[0]])
                # print(data.state[ind[0]])
            # print(torch.cat([value.unsqueeze(1),data.value_root.unsqueeze(1)],1))
            # print(data.prob.shape,prob.shape)
            # lss1 = ((torch.log(data.prob)-torch.log(prob))*data.prob).sum(1).mean()
            lss1 = kl_loss(torch.log(prob),data.prob)
            # print(data.var[0],torch.cat([prob[0:1,:],data.prob[0:1,:]],0).T)
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
    
        


    

        
        
    
