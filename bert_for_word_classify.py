from Bert_finetune import Bert_word_pre
from data_process import creat_batch_for_wordpre,general_transform_text2list,generate_vocab_normalway,word_pre_load
from torch.utils.data import DataLoader
from Config_load import *
import torch


json2list=general_transform_text2list("data/chinese-poetry/chuci/chuci.json",type="json",args=['content'])
data=json2list.getdata()
# transform list to token
list2token=generate_vocab_normalway(data,map_dir="words_info.json")
sentences,token_list,idx2word,word2idx,vocab_size=list2token.transform()
batch = creat_batch_for_wordpre(100,word2idx,token_list,maxlen=maxlen)
loader = DataLoader(word_pre_load(batch), batch_size, True)

model=Bert_word_pre(n_layers=n_layers,
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
        model.display(batch=loader, load_dir="checkpoint/checkpoint_199.pth",map_dir="words_info.json")
