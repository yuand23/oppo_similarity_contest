import torch
# from transformers import AlbertConfig, AlbertModel
from transformers import BertConfig, BertModel
import numpy as np
from torch.utils import data
from sklearn.model_selection import train_test_split
import pandas as pd
import sys
from collections import Counter
import os
import json
from torch.utils.data import TensorDataset
from torch import nn
from transformers import AdamW, get_linear_schedule_with_warmup
from torch.optim.lr_scheduler import StepLR
import math

min_count = 5
maxlen = 32
batch_size = 64
epoches = 20
has_extra_feature = False
learning_rate = 1e-5
dict_path = '/home/juan.du/nlp/bert_models/bert-base-chinese/vocab.txt'
# dict_path = '/home/juan.du/nlp/sentiment_cls/models/albert_chinese_tiny/vocab.txt'
pretrained_model = "/home/juan.du/nlp/bert_models/bert-base-chinese/"
# pretrained_model = "/home/juan.du/nlp/sentiment_cls/models/albert_chinese_tiny/"
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def load_vocab(dict_path, encoding='utf-8', simplified=False, startswith=None):
    """从bert的词典文件中读取词典
    """
    token_dict = {}
    with open(dict_path, encoding=encoding) as reader:
        for line in reader:
            token = line.split()
            token = token[0] if token else line.strip()
            token_dict[token] = len(token_dict)

    if simplified:  # 过滤冗余部分token
        new_token_dict, keep_tokens = {}, []
        startswith = startswith or []
        for t in startswith:
            new_token_dict[t] = len(new_token_dict)
            keep_tokens.append(token_dict[t])

        for t, _ in sorted(token_dict.items(), key=lambda s: s[1]):
            if t not in new_token_dict:
                keep = True
                if len(t) > 1:
                    for c in Tokenizer.stem(t):
                        if (
                            Tokenizer._is_cjk_character(c) or
                            Tokenizer._is_punctuation(c)
                        ):
                            keep = False
                            break
                if keep:
                    new_token_dict[t] = len(new_token_dict)
                    keep_tokens.append(token_dict[t])

        return new_token_dict, keep_tokens
    else:
        return token_dict

def truncate_sequences(maxlen, index, *sequences):
    """截断总长度至不超过maxlen
    """
    sequences = [s for s in sequences if s]
    while True:
        lengths = [len(s) for s in sequences]
        if sum(lengths) > maxlen:
            i = np.argmax(lengths)
            sequences[i].pop(index)
        else:
            return sequences

def load_data(filename, data_typ = 'train'):
    """加载数据
    单条格式：(文本1 ids, 文本2 ids, 标签id)
    data_typ = 'train','dev' or 'test'
    """
    D = []
    with open(filename) as f:
        for l in f:
            l = l.strip().split('\t')
            if len(l) == 3:
                a, b, c = l[0], l[1], int(l[2])
            elif data_typ == 'test' and len(l) == 2:
                a, b = l[0], l[1]
            else:
            	continue # 未标注数据剔除
                # a, b, c = l[0], l[1], -5  # 未标注数据，标签为-5
            a = [int(i) for i in a.split(' ')]
            b = [int(i) for i in b.split(' ')]
            truncate_sequences(maxlen, -1, a, b)
            if data_typ in ['train','dev']:
                D.append((a, b, c)) 
            else:
                D.append((a, b))
    return D

def sample_convert(text1, text2, dic, freq_dic, label=[], random=False):
    """bert # 0: pad, 100: unk, 101: cls, 102: sep, 103: mask, 5: no, 6: yes
    # 0: pad, 1: unk, 2: cls, 3: sep, 4: mask, 5: no, 6: yes
    # dic为{token_id:bert_token_id}
    # freq_dic为{token_id:freq}
    """
    # co occurence score
    co_occurence = set(text1).intersection(set(text2))
    score_feature = sum([1/math.exp(freq_dic[i]) for i in co_occurence]) # score = sum of 1/e^freq
    text1_ids = [dic.get(t, 100) for t in text1]
    text2_ids = [dic.get(t, 100) for t in text2]

    token_ids = [101] + text1_ids + [102] + text2_ids + [102]
    segment_ids = [0] * len(token_ids)
    mask_ids = [1] * len(token_ids)
    return token_ids, segment_ids, mask_ids, score_feature, label

def pad(lst):
    pad_len = maxlen-len(lst) # 限制最大长度
    return lst[:maxlen] + [0]*pad_len
    
def create_dataset(data_lst, dic, freq_dic, data_typ='train'): # data_lst is from function load_data
    LT = torch.LongTensor
    token_lst, seg_lst, mask_lst, score_feature_lst, label_lst = [],[],[],[],[]
    if data_typ in ['train','dev']:
        for s1,s2,lb in data_lst:
            token_ids, seg_ids, mask_ids, score_feature, label = sample_convert(s1,s2,dic,freq_dic,lb)
            token_lst.append(pad(token_ids))
            seg_lst.append(pad(seg_ids))
            mask_lst.append(pad(mask_ids))
            score_feature_lst.append(score_feature)
            label_lst.append(label)
    else:
        for s1,s2 in data_lst:
            token_ids, seg_ids, mask_ids, score_feature, label = sample_convert(s1,s2,dic,freq_dic)
            token_lst.append(pad(token_ids))
            seg_lst.append(pad(seg_ids))
            mask_lst.append(pad(mask_ids))
            score_feature_lst.append(score_feature)
            label_lst.append([-5]) # fake label
    token_tensor = LT(token_lst)
    seg_tensor = LT(seg_lst)
    mask_tensor = LT(mask_lst)
    extra_feature = torch.FloatTensor(score_feature_lst)
    score_feature_tensor = (extra_feature - torch.min(extra_feature))/(torch.max(extra_feature) - torch.min(extra_feature)) # 特征归一化
    label_tensor = LT(label_lst)
    dt = TensorDataset(token_tensor, mask_tensor, seg_tensor, score_feature_tensor, label_tensor)
    return dt

class AlbertClassfier(torch.nn.Module):
    # def __init__(self, abert_model, config, num_class=2):
    def __init__(self, bert_model, config):
        super(AlbertClassfier, self).__init__()
        self.bert = bert_model
        self.fc1 = torch.nn.Linear(config.hidden_size, 128)
        self.fc2 = torch.nn.Linear(128, 2)
        self.dropout = nn.Dropout(0.5)
        self.relu = torch.nn.ReLU()
        # self.softmax = torch.nn.LogSoftmax(dim=1)
    def forward(self, input_ids, attn_masks, token_type_ids, return_hidden=False): 
        # albert_out = self.albert_model(kwargs['input_ids'], kwargs['attn_masks'], kwargs['token_type_ids']).last_hidden_state[:,0,:] #Sentence vector [batch_size, hidden_size]
        albert_out = self.bert(input_ids, attn_masks, token_type_ids).last_hidden_state[:,0,:] #Sentence vector [batch_size, hidden_size]
        # print(torch.Tensor.float(albert_out).mean(0))
        # print(torch.Tensor.float(albert_out).std(0))
        # sys.exit()
        last_hidden = self.fc1(albert_out)
        bert_out = self.relu(last_hidden)
        bert_out = self.dropout(bert_out)
        albert_out = self.fc2(last_hidden)
        if return_hidden:
            return last_hidden
        else:
            return albert_out
        
class Linear_cls(torch.nn.Module):
    def __init__(self, cls_model, albert_config, num_class=2):
        super(Linear_cls, self).__init__()
        self.cls_model = cls_model
        self.cls_with_extra_feature = torch.nn.Linear(128+1, num_class) # one feature by default
    # def forward(self, **kwargs): 
    def forward(self, input_ids, attn_masks, token_type_ids, extra_feature): # extra_fea: [batch_size, 1]
        # Sentence vector [batch_size, hidden_size]
        out = self.cls_model(input_ids, attn_masks, token_type_ids, return_hidden=True)
        # print(torch.Tensor.float(albert_out).mean(0))
        # print(torch.Tensor.float(albert_out).std(0))
        # sys.exit()
        extra_feature = torch.unsqueeze(extra_feature, 1) # [batch_size]变成[batch_size,1]
        out = torch.cat((out, extra_feature), 1)
        out = self.cls_with_extra_feature(out)
        return out
            
if __name__ == '__main__':
    model_dir = './models/'
    train_data = './data/gaiic_track3_round1_train_20210228.tsv'
    # 加载数据集
    all_data = load_data(
        './data/gaiic_track3_round1_train_20210228.tsv',
        data_typ = 'train'
    )
    test_data = load_data(
        './data/gaiic_track3_round1_testA_20210228.tsv',
        data_typ = 'test'
    )
    # split all_data
    train_data = [d for i, d in enumerate(all_data) if i % 9 != 0]
    valid_data = [d for i, d in enumerate(all_data) if i % 9 == 0]

    # 统计词频
    tokens = {}
    cnt_all = 0
    for d in all_data + test_data:
        for i in d[0] + d[1]:
            tokens[i] = tokens.get(i, 0) + 1
            cnt_all += 1

    # 保存每个token的频率数据，token_id为给定数据集中的编号
    tokens_freq = {k:round(v/float(cnt_all),6) for k,v in tokens.items()}
    with open('./models/token_freq.json', 'w') as f1:
        json.dump(tokens_freq, f1)
        
    tokens = {i: j for i, j in tokens.items() if j >= min_count}  # {token:cnt}
    token_high_freq_ids = [
        i for i, j in sorted(tokens.items(), key=lambda s: -s[1])
    ]
    
    # BERT词频
    counts = json.load(open('counts.json')) # {token:freq}
    del counts['[CLS]']
    del counts['[SEP]']
    token_dict = load_vocab(dict_path)
    bert_freqs = [
        counts.get(i, 0) for i, j in sorted(token_dict.items(), key=lambda s: s[1])
    ]
    keep_tokens = list(np.argsort(bert_freqs)[::-1])
    bert_tokens = keep_tokens[:len(tokens)] # [token_id] ordered by desc freq
    
    dic_to_bert_id = dict(zip(token_high_freq_ids, [int(i) for i in bert_tokens]))
    
    # 数据集token_id 和 bert中id 对应词典save dict to file {id:bert_id}
    with open('./models/dic_to_bert_id.json', 'w') as f2:
        json.dump(dic_to_bert_id, f2)
        
    # 生成数据
    train_dataset = create_dataset(train_data, dic_to_bert_id, tokens_freq, data_typ = 'train')
    train_iter = data.DataLoader(dataset=train_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)
    dev_dataset = create_dataset(valid_data, dic_to_bert_id, tokens_freq, data_typ = 'dev')
    dev_iter = data.DataLoader(dataset=dev_dataset,
                                batch_size=batch_size,
                                shuffle=True,
                                num_workers=4)                                

    pretrained_bert_model = BertModel.from_pretrained(pretrained_model)
    # pretrained_bert_model = AlbertModel.from_pretrained(pretrained_model)
    bert_config = BertConfig.from_pretrained(pretrained_model)
    # bert_config = AlbertConfig.from_pretrained(pretrained_model)

    if has_extra_feature: 
        model = AlbertClassfier(pretrained_bert_model, bert_config)
        model.load_state_dict(torch.load('./models/best_model.dic')) 
        model = Linear_cls(model, bert_config)
        # 只调最后一层分类器
        for cnme, child in model.named_children():
            for nme, param in child.named_parameters():
                if cnme.find('cls_with_extra_feature') > -1:
                    param.requires_grad = True
                else:
                    param.requires_grad = False
    else:
        model = AlbertClassfier(pretrained_bert_model, bert_config)
        model.load_state_dict(torch.load('./models/best_model_bert.dic')) 
    model.to(device) 
    # print(model)
    # sys.exit()  
    # train
    criterion = nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    # num_training_steps = int(len(train_dataset)/batch_size)*epoches
    # print("num_training_steps: ",num_training_steps)
    # scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=num_training_steps)
    scheduler = StepLR(optimizer, step_size=10, gamma=0.1)
    best_dev_score = 0.0
    
    for epoch in range(epoches):
        train_loss_sum = 0.0
        train_accu = 0
        model.train()        
        for step, batch in enumerate(train_iter):
            # token_ids, attn_mask, segment_mask, labels = batch
            token_ids, attn_mask, segment_mask, score_feature, labels = batch
            labels = labels.to(device)
            score_feature = score_feature.to(device)
            # print(token_ids, attn_mask, labels)
            if not has_extra_feature: # 参数全调
                inputs = {'input_ids' : token_ids.to(device), 
                'attn_masks' : attn_mask.to(device), 
                'token_type_ids': segment_mask.to(device)} 
            else:
            	inputs = {'input_ids' : token_ids.to(device), 
                'attn_masks' : attn_mask.to(device), 
                'token_type_ids': segment_mask.to(device),
                'extra_feature': score_feature.to(device)}
            out = model(**inputs)
            loss = criterion(out, labels)
            loss.backward() # Back propagation
            optimizer.step() # Gradient update
            train_loss_sum += loss.cpu().data.numpy()
            train_accu += (out.argmax(1) == labels).sum().cpu().data.numpy()
            optimizer.zero_grad()
            if step % 100 == 0:
                print("train loss: %f, acc: %f" % (loss.item(), train_accu/(batch_size*(step+1))))
        scheduler.step()
        dev_loss_sum=0.0
        dev_accu=0
        model.eval()
        for step, batch in enumerate(dev_iter):
            token_ids, attn_mask, segment_mask, score_feature, labels = batch
            labels = labels.to(device)
            score_feature = score_feature.to(device)
            with torch.no_grad():
                if not has_extra_feature: # 参数全调
                    inputs = {'input_ids' : token_ids.to(device), 
                    'attn_masks' : attn_mask.to(device), 
                    'token_type_ids': segment_mask.to(device)} 
                else:
                	inputs = {'input_ids' : token_ids.to(device), 
                    'attn_masks' : attn_mask.to(device), 
                    'token_type_ids': segment_mask.to(device),
                    'extra_feature': score_feature.to(device)}
                out = model(**inputs)
                loss = criterion(out,labels)
                dev_loss_sum += loss.cpu().data.numpy()
                dev_accu += (out.argmax(1)==labels).sum().cpu().data.numpy()    
        print("epoch % d,train loss:%f,train acc:%f,dev loss:%f,dev acc:%f" %
        (epoch+1,train_loss_sum/len(train_dataset),train_accu/len(train_dataset),dev_loss_sum/len(dev_dataset),dev_accu/len(dev_dataset)))
        dev_score = dev_accu/len(dev_dataset)
        if dev_score > best_dev_score:
            best_val_score = dev_score
            if not has_extra_feature:
                torch.save(model.state_dict(), './models/best_model.dic')
            else:
                torch.save(model.state_dict(), './models/best_model_extra_feature.dic')
    # sys.exit()
else:
    """预测结果到文件
    """
    out_file = './output/result.txt'
    pretrained_bert_model = BertModel.from_pretrained(pretrained_model)
    bert_config = BertConfig.from_pretrained(pretrained_model)
    if has_extra_feature: 
        model = AlbertClassfier(pretrained_bert_model, bert_config)
        model.load_state_dict(torch.load('./models/best_model.dic')) 
        model = Linear_cls(model, bert_config)
    else:
        model = AlbertClassfier(pretrained_bert_model, bert_config)
    model.to(device)
    
    with open('./models/dic_to_bert_id.json') as f:
        d = json.load(f)
    dic_to_bert_id = {int(k):v for k,v in d.items()}
    with open('./models/token_freq.json') as f:
        d = json.load(f)
    tokens_freq = {int(k):v for k,v in d.items()}
        
    test_data = load_data(
        './data/gaiic_track3_round1_testA_20210228.tsv',
        data_typ = 'test'
    )
    test_dataset = create_dataset(test_data, dic_to_bert_id, tokens_freq, data_typ = 'test')
    test_iter = data.DataLoader(dataset=test_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                num_workers=4)   
    F = open(out_file, 'w')
    model.eval()
    softmax = nn.Softmax(dim=1)
    for step, batch in enumerate(test_iter):
        token_ids, attn_mask, segment_mask, score_feature, labels = batch
        with torch.no_grad():
            if not has_extra_feature: # 参数全调
                inputs = {'input_ids' : token_ids.to(device), 
                'attn_masks' : attn_mask.to(device), 
                'token_type_ids': segment_mask.to(device)} 
            else:
            	inputs = {'input_ids' : token_ids.to(device), 
                'attn_masks' : attn_mask.to(device), 
                'token_type_ids': segment_mask.to(device),
                'extra_feature': score_feature.to(device)}
            out = model(**inputs)
            # out = model(token_ids.to(device))
            # print(out) 
            # print(out.argmax(1))
            result = softmax(out)[:,1].tolist()
            # sys.exit()
        for p in result:
            F.write('%f\n' % p)
    F.close()
      