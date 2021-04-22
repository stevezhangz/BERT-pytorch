# encode utf-8
# code by steve zhang z
# Time: 4/22/2021
# electric address: stevezhangz@163.com
from bert import *
import torch
import torch.nn as nn
import torch.optim as optim
from Config_load import *
from data_process import *
import random


np.random.seed(random_seed)


# transform json to list
#json2list=general_transform_text2list("data/demo.txt",type="txt")
json2list=general_transform_text2list("data/chinese-poetry/chuci/chuci.json",type="json",args=['content'])
data=json2list.getdata()
# transform list to token
list2token=generate_vocab_normalway(data,map_dir="words_info.json")
sentences,token_list,idx2word,word2idx,vocab_size=list2token.transform()
batch = creat_batch(batch_size,max_pred,maxlen,word2idx,idx2word,token_list,0.15)
loader = Data.DataLoader(Text_file(batch), batch_size, True)

model=Bert(n_layers=n_layers,
                 vocab_size=vocab_size,
                 emb_size=d_model,
                 max_len=maxlen,
                 seg_size=n_segments,
                 dff=d_ff,
                 dk=d_k,
                 dv=d_v,
                 n_head=n_heads,
                 n_class=2,
                 drop=drop)

if use_gpu:
    with torch.cuda.device(device) as device:
        model.to(device)
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adadelta(model.parameters(), lr=lr)
        model.Train_for_mask_guess(epoches=epoches,
                    train_data_loader=loader,
                    optimizer=optimizer,
                    criterion=criterion,
                    save_dir=weight_dir,
                    save_freq=100,
                    load_dir="checkpoint/checkpoint_199.pth",
                    use_gpu=use_gpu,
                    device=device
                    )
else:
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adadelta(model.parameters(), lr=lr)
    model.Train_for_mask_guess(epoches=epoches,
                train_data_loader=loader,
                optimizer=optimizer,
                criterion=criterion,
                save_dir=weight_dir,
                save_freq=50,
                load_dir="checkpoint/checkpoint_199.pth",
                use_gpu=use_gpu,
                device=device
                )
