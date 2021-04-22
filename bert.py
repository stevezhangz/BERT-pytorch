# encode utf-8
# code by steve zhang z
# Time: 4/22/2021
# electric address: stevezhangz@163.com
import torch
from torch import nn
import numpy as np
import os
import argparse
import math


class Grelu(nn.Module):
    def __init__(self):
        super(Grelu, self).__init__()

    def forward(self,x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))

def grelu(x):
    # GAUSSIANERRORLINEARUNITS(GELUS)
    # url: https://arxiv.org/pdf/1606.08415.pdf
    return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class Embedding(nn.Module):
    def __init__(self,
                 vocab_size,
                 emb_size,
                 max_len,
                 seg_size):
        super(Embedding, self).__init__()
        self.emb_x=nn.Embedding(vocab_size,emb_size)
        self.emb_pos=nn.Embedding(max_len,emb_size)
        self.emb_seg=nn.Embedding(seg_size,emb_size)
        # I dont know why use norm to proces bedding data, I guess that it's help to convergence, perhaps.
        self.norm=nn.LayerNorm(emb_size)
    def forward(self,x,seg):
        length=x.size(1)
        pos=torch.arange(length)
        return self.norm(self.emb_x(x)+self.emb_pos(pos)+self.emb_seg(seg))

class multi_head(nn.Module):
    def __init__(self,
                 emb_size,
                 dk,
                 dv,
                 n_head,
                 ):
        super(multi_head, self).__init__()
        self.Q=nn.Linear(emb_size,n_head*dk)
        self.K=nn.Linear(emb_size,n_head*dk)
        self.V=nn.Linear(emb_size,n_head*dv)
        self.layer_norm=nn.LayerNorm(emb_size)
        self.Linear=nn.Linear(n_head*dk,emb_size)
        self.n_head=n_head
        self.dk=dk
        self.dv=dv

    def dot_product_with_musk(self,query,key,value,mask):
        dotproduct=torch.matmul(query,key.transpose(-1,-2))/np.sqrt(self.dk)
        dotproduct=dotproduct.masked_fill_(mask,1e-9)
        # size: batch_size, n_head,length,length
        return torch.matmul(nn.Softmax(dim=-1)(dotproduct),value)

    def forward(self,Input,mask):
        residual=Input
        batch_size=Input.size()[0]
        # Size:  batch_size, n_head,seq_length,dk(or dv)
        K,Q,V=self.K(Input).view(batch_size,self.n_head,-1,self.dk),\
              self.Q(Input).view(batch_size,self.n_head,-1,self.dk),\
              self.V(Input).view(batch_size,self.n_head,-1,self.dv)
        # transform original type of mask into multi-head type
        try:
            mask=mask.unsquieeze(1).repeat(1,self.n_head,1,1)
        except:
            mask = mask.data.unsqueeze(1).repeat(1, self.n_head, 1, 1)
        context=self.dot_product_with_musk(query=Q,
                                           key=K,
                                           value=V,
                                           mask=mask
                                           )
        context=context.transpose(1,2).contiguous().view(batch_size,-1,self.n_head*self.dk)
        # now shape of context could be defined as : batch_size,length, n_head*length
        output=self.Linear(context)
        # finally return batch_size,length,emb_size
        return self.layer_norm(output+residual)

class basic_block(nn.Module):
    def __init__(self,emb_size,
                 dff,
                 dk,
                 dv,
                 n_head):
        super(basic_block, self).__init__()
        self.shit_forward=nn.Sequential(
            nn.Linear(emb_size,dff),
            nn.Linear(dff,emb_size)
        )
        self.multi_head=multi_head(emb_size,dk,dv,n_head)

    def forward(self,Input,mask):
        return self.shit_forward(grelu(self.multi_head(Input,mask)))


class Bert(nn.Module):
    def __init__(self,
                 n_layers,
                 vocab_size,
                 emb_size,
                 max_len,
                 seg_size,
                 dff,
                 dk,
                 dv,
                 n_head,
                 n_class):
        super(Bert, self).__init__()
        self.vocab_size=vocab_size
        self.emb_size=emb_size
        self.emb_layer=Embedding(vocab_size,emb_size,max_len,seg_size)
        self.encoder_layer=nn.Sequential(*[basic_block(emb_size,dff,dk,dv,n_head) for i in range(n_layers)])
        self.fc1=nn.Sequential(
            nn.Linear(emb_size, vocab_size),
            nn.Dropout(0.5),
            nn.Tanh(),
            nn.Linear(vocab_size, n_class)
        )
        fc2=nn.Linear(emb_size, vocab_size)
        fc2.weight=self.emb_layer.emb_x.weight
        self.fc2=nn.Sequential(
            nn.Linear(emb_size, emb_size),
            Grelu(),
            fc2
        )
    def get_mask(self,In):
        batch_size,length,mask=In.size()[0],In.size()[1],In
        mask=mask.eq(0).unsqueeze(1)
        return mask.data.expand(batch_size,length,length)

    def forward(self,x,seg,mask_):
        mask=self.get_mask(x)
        output=self.emb_layer(x=x,seg=seg)
        for layer in self.encoder_layer:
            output=layer(output,mask)
        cls=self.fc1(output[:,0])
        masked_pos = mask_[:, :, None].expand(-1, -1,self.emb_size)
        masked=torch.gather(output,1,masked_pos)
        logits=self.fc2(masked)
        return logits,cls

    def Train(self,epoches,criterion,optimizer,train_data_loader,use_gpu,device,
              eval_data_loader=None,save_dir="./checkpoint",load_dir=None,save_freq=5,
              ):
        import tqdm
        if load_dir!=None:
            if os.path.exists(load_dir):
                checkpoint=torch.load(load_dir)
                try:
                    self.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except:
                    print("fail to load the state_dict")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for epc in  range(epoches):
            tq=tqdm.tqdm(train_data_loader)
            for seq,(input_ids, segment_ids, masked_tokens, masked_pos, isNext) in enumerate(tq):
                if use_gpu:
                    input_ids, segment_ids, masked_tokens, masked_pos, isNext=input_ids.to(device), \
                                                                              segment_ids.to(device), \
                                                                              masked_tokens.to(device), \
                                                                              masked_pos.to(device),\
                                                                              isNext.to(device)
                logits_lm, logits_clsf = self(x=input_ids, seg=segment_ids, mask_=masked_pos)
                loss_word = criterion(logits_lm.view(-1, self.vocab_size), masked_tokens.view(-1))  # for masked LM
                loss_word = (loss_word.float()).mean()
                loss_cls = criterion(logits_clsf, isNext)  # for sentence classification
                loss = loss_word + loss_cls
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                tq.set_description(f"train Epoch {epc+1}, Batch{seq}")
                tq.set_postfix(train_loss=loss)

            if eval_data_loader!=None:
                tq=tqdm.tqdm(eval_data_loader)
                with torch.no_grad():
                    for seq,(input_ids, segment_ids, masked_tokens, masked_pos, isNext) in enumerate(tq):
                        input_ids, segment_ids, masked_tokens, masked_pos, isNext = input_ids.to(device), \
                                                                                    segment_ids.to(device), \
                                                                                    masked_tokens.to(device), \
                                                                                    masked_pos.to(device), \
                                                                                    isNext.to(device)
                        logits_lm, logits_clsf = self(x=input_ids, seg=segment_ids, mask_=masked_pos)
                        loss_word = criterion(logits_lm.view(-1, self.vocab_size),
                                              masked_tokens.view(-1))  # for masked LM
                        loss_word = (loss_word.float()).mean()
                        loss_cls = criterion(logits_clsf, isNext)  # for sentence classification
                        loss = loss_word + loss_cls
                        tq.set_description(f"eval Epoch {epc + 1}, Batch{seq}")
                        tq.set_postfix(train_loss=loss)

            if (epc+1)%save_freq==0:
                checkpoint = {'epoch': epc,
                              'best_loss': criterion,
                              'model': self.state_dict(),
                              'optimizer': optimizer.state_dict()
                              }
                torch.save(checkpoint, save_dir+ f"/checkpoint_{epc}.pth")














