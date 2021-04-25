from bert import *
from Config_load import *



class Bert_word_pre(nn.Module):
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
                 n_class,
                 drop):
        super(Bert_word_pre, self).__init__()
        self.vocab_size=vocab_size
        self.emb_size=emb_size
        self.emb_layer=Embedding(vocab_size,emb_size,max_len,seg_size)
        self.encoder_layer=nn.Sequential(*[basic_block(emb_size,dff,dk,dv,n_head) for i in range(n_layers)])
        self.fc1=nn.Sequential(
            nn.Linear(emb_size, vocab_size),
            nn.Dropout(drop),
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

    def forward(self,x):
        mask=self.get_mask(x)
        output=self.emb_layer(x=x,seg=None)
        for layer in self.encoder_layer:
            output=layer(output,mask)
        cls=self.fc2(output[:,1:])
        return cls

    def display(self,batch,load_dir,map_dir):
        import json
        if load_dir != None:
            if os.path.exists(load_dir):
                checkpoint = torch.load(load_dir)
                try:
                    self.load_state_dict(checkpoint['model'])
                except:
                    print("fail to load the state_dict")
        map_file=json.load(open(map_dir,"r"))["idx2word"]
        for x in batch:
            pre=self(x)
            pre=pre.data.max(2)[1][0].data.numpy()
            transform=[]
            for i in pre:
                try:
                    word_pre=map_file[int(i)]
                except:
                    word_pre="mistake"
                transform.append(word_pre)
            print("prediction_words:",transform)
            print("prediction_token:", pre)


    def Train(self, epoches, criterion, optimizer, train_data_loader, use_gpu, device,
              eval_data_loader=None, save_dir="./checkpoint", load_dir=None, save_freq=5,
              ):
        import tqdm
        if load_dir != None:
            if os.path.exists(load_dir):
                checkpoint = torch.load(load_dir)
                try:
                    self.load_state_dict(checkpoint['model'])
                    optimizer.load_state_dict(checkpoint['optimizer'])
                except:
                    print("fail to load the state_dict")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        for epc in range(epoches):
            tq = tqdm.tqdm(train_data_loader)
            for seq, (input_ids, classi) in enumerate(tq):
                if use_gpu:
                    input_ids, classi = input_ids.to(device), classi.to(device)
                logits_clsf = self(x=input_ids)
                loss_cls = criterion(logits_clsf, classi)
                optimizer.zero_grad()
                loss_cls.backward()
                optimizer.step()
                tq.set_description(f"train Epoch {epc + 1}, Batch{seq}")
                tq.set_postfix(train_loss=loss_cls)

            if eval_data_loader != None:
                tq = tqdm.tqdm(eval_data_loader)
                with torch.no_grad():
                    for epc in range(epoches):
                        tq = tqdm.tqdm(train_data_loader)
                        for seq, (input_ids, classi) in enumerate(tq):
                            if use_gpu:
                                input_ids, classi = input_ids.to(device), classi.to(device)
                            logits_clsf = self(x=input_ids)
                            loss_cls = criterion(logits_clsf, classi)
                            optimizer.zero_grad()
                            loss_cls.backward()
                            optimizer.step()
                            tq.set_description(f"Eval Epoch {epc + 1}, Batch{seq}")
                            tq.set_postfix(train_loss=loss_cls)

            if (epc + 1) % save_freq == 0:
                checkpoint = {'epoch': epc,
                              'best_loss': criterion,
                              'model': self.state_dict(),
                              'optimizer': optimizer.state_dict()
                              }
                torch.save(checkpoint, save_dir + f"/checkpoint_{epc}.pth")

class Bert_classify(nn.Module):
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
                 n_class,
                 drop):
        super(Bert_classify, self).__init__()
        self.vocab_size=vocab_size
        self.emb_size=emb_size
        self.emb_layer=Embedding(vocab_size,emb_size,max_len,seg_size)
        self.encoder_layer=nn.Sequential(*[basic_block(emb_size,dff,dk,dv,n_head) for i in range(n_layers)])
        self.fc1=nn.Sequential(
            nn.Linear(emb_size, vocab_size),
            nn.Dropout(drop),
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

    def forward(self,x):
        mask=self.get_mask(x)
        output=self.emb_layer(x=x,seg=None)
        for layer in self.encoder_layer:
            output=layer(output,mask)
        cls=self.fc1(output[:,0])
        return cls

    def display(self,batch,load_dir):
        if load_dir != None:
            if os.path.exists(load_dir):
                checkpoint = torch.load(load_dir)
                try:
                    self.load_state_dict(checkpoint['model'])
                except:
                    print("fail to load the state_dict")
                for i in batch:
                    logits_clsf = self(x=i)
                    print(logits_clsf)


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
            for seq,(input_ids,classi) in enumerate(tq):
                if use_gpu:
                     input_ids, classi=input_ids.to(device), classi.to(device)
                logits_clsf = self(x=input_ids)
                loss_word = criterion(logits_clsf.view(-1, self.vocab_size), logits_clsf.view(-1))
                loss_word = (loss_word.float()).mean()
                optimizer.zero_grad()
                loss_word.backward()
                optimizer.step()
                tq.set_description(f"train Epoch {epc+1}, Batch{seq}")
                tq.set_postfix(train_loss=loss_word)

            if eval_data_loader!=None:
                tq=tqdm.tqdm(eval_data_loader)
                with torch.no_grad():
                    for epc in range(epoches):
                        tq = tqdm.tqdm(train_data_loader)
                        for seq, (input_ids, classi) in enumerate(tq):
                            if use_gpu:
                                input_ids, classi = input_ids.to(device), classi.to(device)
                            logits_clsf = self(x=input_ids)
                            loss_word = criterion(logits_clsf.view(-1, self.vocab_size), logits_clsf.view(-1))
                            loss_word = (loss_word.float()).mean()
                            optimizer.zero_grad()
                            loss_word.backward()
                            optimizer.step()
                            tq.set_description(f"Eval Epoch {epc + 1}, Batch{seq}")
                            tq.set_postfix(train_loss=loss_word)

            if (epc+1)%save_freq==0:
                checkpoint = {'epoch': epc,
                              'best_loss': criterion,
                              'model': self.state_dict(),
                              'optimizer': optimizer.state_dict()
                              }
                torch.save(checkpoint, save_dir+ f"/checkpoint_{epc}.pth")



