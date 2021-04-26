# encode utf-8
# code by steve zhang z
# Time: 4/22/2021
# electric address: stevezhangz@163.com
import re
from random import *
import torch.utils.data as Data
import os
import json
import thulac
import numpy as np
import torch
class general_transform_text2list:
    """
    notification: All series of data process method here only support the list type sentences, so whether json or txt file
    should be transformed into list type, such as [s1,s2,s3,s4,s5]

    """
    def __init__(self,text_dir,args=[],type="json"):
        assert type in ["json","txt"], print("plz give a txt or a json")
        self.text=text_dir
        self.type=type
        self.arg=args

    def getdata(self):
        if self.type=="json":
            return self.for_json()
        elif self.type=="txt":
            return self.for_txt()
        else:
            raise KeyError
    def for_txt(self):
        sentences=[]
        if os.path.exists(self.text):
            with open(self.text,"r") as f:
                for i in f:
                    sentences.append(i)
        return sentences
    def for_json(self):
        sentences=[]
        keys=[i for i in self.arg]
        with open(self.text,"r") as f:
            f=json.load(f)
            for i in f:
                for key in keys:
                    if isinstance(i[key],list):
                        for j in i[key]:
                            sentences.append(j)
                    elif isinstance(i[key],str):
                        sentences.append(i[key])
        return sentences


class generate_vocab_normalway:
    """
    :notification: before using this method please transform the texts into the form of [s1,s2,s3,s4,s5,s6....], which
        "s" represents a individual sentence.
    :param: 1. text_list(transformed text files)
            2. map_dir(throughout this method, u will obtained the bijection between words and its ids, all of them saved
            in the map_dir, so this method have to update it when process different tasks)(except that, the map_dir
            contains three keys: words, word2idx and idx2word, which correspond to lib of words, map from word to id as
            well as map from id to word)
    """
    def __init__(self,text_list,map_dir,record_update=True,language="Chinese"):
        self.text_list=text_list
        self.map_dir=map_dir
        self.language=language
        self.update=record_update

    def transform(self):
        if os.path.exists(self.map_dir):
            with open(self.map_dir, "r") as file_:
                map_file = json.load(file_)
                use_before = 1
                file_.close()
        else:
            use_before = 0
        cut = thulac.thulac()
        if use_before:
            words = map_file["words"]
        else:
            words = []
        sentences = []
        for i in self.text_list:
            sentence = re.sub("[.,!?，。:：\n\\-]", '', i.lower())
            if self.language=="Chinese":
                sentence = list(set(np.array(cut.cut(sentence, text=False))[:, 0]))
            elif self.language=="English":
                sentence=list(set(sentence.split(" ")))
            sentences.append(sentence)
            for i in sentence:
                if use_before:
                    if i not in map_file["words"]:
                        words.append(i)
                else:
                    words.append(i)
        words = list(set(words))
        if use_before:
            word2idx = map_file["word2idx"]
        else:
            word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
        for seq, val in enumerate(words):
            word2idx[val] = seq + 4
        id_sentence = []
        for i in sentences:
            id_sentence.append([word2idx[j] for j in i])
        vocab_size = len(word2idx)
        idx2word = {i: w for i, w in enumerate(word2idx)}

        if self.update:
            if use_before:
                map_file["word2idx"] = word2idx
                map_file["idx2word"] = idx2word
                map_file["words"] = words
                with open(self.map_dir, "w") as file_:
                    json.dump(map_file, file_)
                    file_.close()
            else:
                if self.map_dir == None:
                    map_dir = "word_info.json"
                map_file = {}
                map_file["word2idx"] = word2idx
                map_file["idx2word"] = idx2word
                map_file["words"] = words
                with open(map_dir, "w") as f:
                            json.dump(map_file, f)
        return sentences, id_sentence, idx2word, word2idx, vocab_size


def generate_vocab_from_poem_chuci(poem_dir,map_dir):
    """
    :poem introduction: This poem was written by Qu Yuan, a great poet in ancient China
    :data link, Thanks: https://codechina.csdn.net/mirrors/chinese-poetry/chinese-poetry?utm_source=csdn_github_accelerator
    :param poem_dir: data/chinese-poetry/chuci/chuci.json
    """
    if os.path.exists(map_dir):
        with open(map_dir,"r") as file_:
            map_file=json.load(file_)
            use_before=1
            file_.close()
    else:
        use_before=0
    cut=thulac.thulac()
    if not os.path.exists(poem_dir):
        raise FileNotFoundError
    else:
        with open(poem_dir, "r") as f:
            json_file = json.load(f)
            if use_before:
                words=map_file["words"]
            else:
                words=[]
            sentences=[]
            for poem in json_file:
                for i in poem["content"]:
                    sentence= re.sub("[.,!?，。:：\\-]", '', i.lower())
                    sentence=list(set(np.array(cut.cut(sentence,text=False))[:,0]))
                    sentences.append(sentence)
                    for i in sentence:
                        if use_before:
                            if i not in map_file["words"]:
                                words.append(i)
                        else:
                            words.append(i)
            words=list(set(words))
            if use_before:
                word2idx =map_file["word2idx"]
            else:
                word2idx = {'[PAD]': 0, '[CLS]': 1, '[SEP]': 2, '[MASK]': 3}
            for seq,val in enumerate(words):
                word2idx[val]=seq+4
            id_sentence=[]
            for i in sentences:
                id_sentence.append([word2idx[j] for j in i])
            vocab_size=len(word2idx)
            idx2word={i:w for i,w in enumerate(word2idx)}
            if use_before:
                map_file["word2idx"]=word2idx
                map_file["idx2word"]=idx2word
                map_file["words"]=words
                with open(map_dir, "w") as file_:
                    json.dump(map_file,file_)
                    file_.close()
            else:
                if map_dir==None:
                    map_dir="word_info.json"
                map_file={}
                map_file["word2idx"] = word2idx
                map_file["idx2word"] = idx2word
                map_file["words"] = words
                with open(map_dir,"w") as f:
                    json.dump(map_file,f)
    return sentences,id_sentence,idx2word,word2idx,vocab_size

def creat_batch_for_wordpre(
                batch_size,
                word2idx,
                token_list,
                maxlen):
    batch=[]
    cnt=0
    if batch_size>len(token_list):
        batch_size=len(token_list)
    while(cnt<batch_size):
        s=choice(token_list)
        In_id=[word2idx['[CLS]']]+s
        if len(In_id)<maxlen:
            n_pad=maxlen-len(In_id)
            In_id.extend([0]*n_pad)
        batch.append(In_id)
        cnt+=1
    return batch

class word_pre_load(Data.Dataset):
    def __init__(self, batch):
        input_ids= torch.LongTensor(batch)
        self.input_ids = input_ids


    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx]
def creat_batch(batch_size,
                max_pred,
                maxlen,
                word2idx,
                idx2word,
                token_list,
                pre_percent):
    """
    here this mechine just have to predict several masked words. and also have to predict whether they are sequential
    :param batch_size:
    :param max_pred:
    :param maxlen:
    :param vocab_size:
    :param word2idx:
    :param token_list:
    :param sentences:
    :return: batch[In_id, seg_id, could_mask, could_mask_tok, isconnect]
    """
    batch=[]
    connect=unconnect=0
    while connect<batch_size/2 or unconnect<batch_size/2:
        s1=choice(token_list)
        s1_index=token_list.index(s1)
        s2 = choice(token_list)
        s2_index=token_list.index(s2)
        In_id=[word2idx['[CLS]']] + s1 + [word2idx['[SEP]']] + s2 + [word2idx['[SEP]']]
        seg_id=[0] * (1 + len(s1) + 1) + [1] * (len(s2) + 1)
        could_mask=[]
        for seq,val in enumerate(In_id):
            if idx2word[val]!='[CLS]' and idx2word[val]!='[SEP]':
                could_mask.append(seq)
        mask_num=min(max_pred,max(int(len(could_mask)*pre_percent),1))
        mask_Inid=np.random.choice(could_mask,int(mask_num))
        mask_pos=[]
        for mIid in mask_Inid:
            In_id[mIid]=word2idx['[MASK]']
            mask_pos.append(mIid)

        pad_need=maxlen-len(In_id)
        In_id.extend([0]*pad_need)
        seg_id.extend([0]*pad_need)
        mask_Inid=[i for i in mask_Inid]
        if mask_num<max_pred:
            mask_Inid.extend([0]*(max_pred-int(mask_num)))
            mask_pos.extend([0]*(max_pred-int(mask_num)))
        if s1_index+1==s2_index and connect<batch_size/2:
            connect+=1
            batch.append([In_id,seg_id,mask_pos,mask_Inid,True])
        if s1_index+1!=s2_index and unconnect<batch_size/2:
            unconnect+=1
            batch.append([In_id, seg_id,mask_Inid,mask_pos, False])
    return batch

def creat_batch_demo(batch_size,max_pred,maxlen,vocab_size,word2idx,token_list,sentences):
    """
    this demo could be found, thanks: https://codechina.csdn.net/mirrors/wmathor/nlp-tutorial/-/tree/master/5-2.BERT
    :param batch_size:
    :param max_pred:
    :param maxlen:
    :param vocab_size:
    :param word2idx:
    :param token_list:
    :param sentences:
    :return:batch
    """
    batch = []
    positive = negative = 0
    while positive != batch_size / 2 or negative != batch_size / 2:
        tokens_a_index, tokens_b_index = randrange(len(sentences)), randrange(
            len(sentences))
        # random choice two sentences
        tokens_a, tokens_b = token_list[tokens_a_index], token_list[tokens_b_index]

        input_ids = [word2idx['[CLS]']] + tokens_a + [word2idx['[SEP]']] + tokens_b + [word2idx['[SEP]']]
        segment_ids = [0] * (1 + len(tokens_a) + 1) + [1] * (len(tokens_b) + 1)
        n_pred = min(max_pred, max(1, int(len(input_ids) * 0.15)))
        cand_maked_pos = [i for i, token in enumerate(input_ids)
                          if token != word2idx['[CLS]'] and token != word2idx['[SEP]']]
        shuffle(cand_maked_pos)
        masked_tokens, masked_pos = [], []
        for pos in cand_maked_pos[:n_pred]:
            masked_pos.append(pos)
            masked_tokens.append(input_ids[pos])
            if random() < 0.8:
                input_ids[pos] = word2idx['[MASK]']
            elif random() > 0.9:
                index = randint(0, vocab_size - 1)
                while index < 4:
                    index = randint(0, vocab_size - 1)
                input_ids[pos] = index
        n_pad = maxlen - len(input_ids)
        input_ids.extend([0] * n_pad)
        segment_ids.extend([0] * n_pad)
        if max_pred > n_pred:
            n_pad = max_pred - n_pred
            masked_tokens.extend([0] * n_pad)
            masked_pos.extend([0] * n_pad)
        if tokens_a_index + 1 == tokens_b_index and positive < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, True])
            positive += 1
        elif tokens_a_index + 1 != tokens_b_index and negative < batch_size / 2:
            batch.append([input_ids, segment_ids, masked_tokens, masked_pos, False])
            negative += 1
    return batch


class Text_file(Data.Dataset):
    def __init__(self, batch):
        input_ids, segment_ids, masked_tokens, masked_pos, isNext = zip(*batch)
        input_ids, segment_ids, masked_tokens, masked_pos, isNext = torch.LongTensor(input_ids),\
                                                                    torch.LongTensor( segment_ids),\
                                                                    torch.LongTensor(masked_tokens),\
                                                                    torch.LongTensor(masked_pos),\
                                                                    torch.LongTensor( isNext)
        self.input_ids = input_ids
        self.segment_ids = segment_ids
        self.masked_tokens = masked_tokens
        self.masked_pos = masked_pos
        self.isNext = isNext

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.segment_ids[idx], self.masked_tokens[idx], self.masked_pos[idx], self.isNext[
            idx]
