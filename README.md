# BERT-pytorch

Repeted by myself, without pre-train.

# How to use

Bash code(preparation)

    sudo apt-get install ipython3
    sudo apt-get install pip
    sudo apt-get install git
    git clone https://github.com/stevezhangz/BERT-pytorch.git
    cd BERT-pytorch
    pip install -r requirements.txt 
    
I prepare two demos for model training(poem and a conversation)
run train_demo.py to train
  
    ipython3 train_demo.py

except that, you have to learn about how to run it on your dataset

(1)first use "general_transform_text2list" in data_process.py to transform txt or json file to list which defined as "[s1,s2,s3,s4.....]"
(2)then use "generate_vocab_normalway" in data_process.py to transform list file to "sentences, id_sentence, idx2word, word2idx, vocab_size"
(3)Last but not least, use "creat_batch" in data_process.py to transform "sentences, id_sentence, idx2word, word2idx, vocab_size" to a batch.
(4)finally using dataloder in pytorch to load data.
for example:

    #json2list=general_transform_text2list("data/demo.txt",type="txt")
    json2list=general_transform_text2list("data/chinese-poetry/chuci/chuci.json",type="json",args=['content'])
    data=json2list.getdata()
    # transform list to token
    list2token=generate_vocab_normalway(data,map_dir="words_info.json")
    sentences,token_list,idx2word,word2idx,vocab_size=list2token.transform()
    batch = creat_batch(batch_size,max_pred,maxlen,vocab_size,word2idx,token_list,sentences)
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
    input_ids, segment_ids, masked_tokens, masked_pos, isNext = \
        torch.LongTensor(input_ids), torch.LongTensor(segment_ids), torch.LongTensor(masked_tokens), \
        torch.LongTensor(masked_pos), torch.LongTensor(isNext)
    loader = Data.DataLoader(Text_file(input_ids, segment_ids, masked_tokens, masked_pos, isNext), batch_size, True)
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
                    )
    if use_gpu:
        with torch.cuda.device(device) as device:
            model.to(device)
            criterion = nn.CrossEntropyLoss()
            optimizer = optim.Adadelta(model.parameters(), lr=lr)
            model.Train(epoches=epoches,
                        train_data_loader=loader,
                        optimizer=optimizer,
                        criterion=criterion,
                        save_dir=weight_dir,
                        save_freq=100,
                        load_dir="checkpoint/checkpoint_199.pth",
                        use_gpu=use_gpu,
                        device=device
                        )


# How to config
Modify super parameters directly in “Config.cfg”

# Pretrain
Because of time, I can't spend time to train the model. You are welcome to use my model for training and contribute pre train weight to this project

# About me
author={
  E-maile:stevezhangz@163.com
}

# Acknowledgement
Acknowledgement for the open-source [poem dataset](https://github.com/chinese-poetry/chinese-poetry)


    
