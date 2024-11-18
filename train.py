from tqdm import tqdm
import pandas as pd
import numpy as np
import re
import pickle
import torch
import torch.nn as nn
import transformers

import os
import torchmetrics
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from transformers import AdamW
import tensorflow.compat.v1 as tf
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score
from transformers import BertTokenizer, BertModel
from IPython.display import FileLink, clear_output
import random
import yaml
import json
import argparse
random.seed(42)

parser = argparse.ArgumentParser(description='Train script')
parser.add_argument('config', type=str, help='Config load')
parser.add_argument('mode', type=str, default='train', help='Mode train or valid')
args = parser.parse_args()


with open(args.config, 'r') as file:
    config = yaml.safe_load(file)
print(config)

model_name = config['model_name']
lang = config['lang']
version = config['version']
length_train = config['length_train']
length_valid = config['length_valid']
max_epochs = config['max_epochs']


train_loader_name = f'data_load/{lang}_train_loader_v{version}_{length_train}.pkl'
val_loader_name = f'data_load/{lang}_val_loader_v{version}_{length_valid}.pkl'
mode = args.mode
sub = ''
checkpoint_way = 'chk_models/BERT{sub}_v{version}_tr{lenth_train}_ep{max_epochs}_{lang}.pkl'
res_way = f'res/BERT{sub}_v{version}_tr{length_train}_ep{max_epochs}_{lang}.json'

from model_list import BERT_v3 as model_module

model = model_module.model(model_name)

def prepare_data(text, tokenizer=BertTokenizer.from_pretrained(model_name)):
    labels_count_1 = 4
    add_element_1 = [1., *[0. for i in range(labels_count_1-1)]]
    labels_count_2 = 3
    add_element_2 = [1., *[0. for i in range(labels_count_2-1)]]
    target_1 = []
    target_2 = []
    mask = []
    features = []
    sub = []
    if isinstance(text, str):
        text = [text]
    for line in tqdm(text):
        tokens = line.strip().split(' ')
        labels_1, tokens = label_input(tokens, tokens.copy(), label='\.', label_id=1, mode='punct', labels_count=labels_count_1)
        labels_1, tokens = label_input(tokens, labels_1, label=',', label_id=2, mode='punct', labels_count=labels_count_1)
        labels_1, tokens = label_input(tokens, labels_1, label='\?', label_id=3, mode='punct', labels_count=labels_count_1)
        labels_1, tokens = label_input(tokens, labels_1, label='##[A-Za-zА-Яа-яйёЙЁ]+|[A-Za-zА-Яа-яйёЙЁ]+', label_id=0, delit=False, mode='punct', labels_count=labels_count_1)
        labels_1 = check_target(labels_1, add_element_1)
        
        labels_2, tokens = label_input(tokens, tokens.copy(), label='[A-ZА-ЯЙЁ]+', label_id=2, delit=False, labels_count=labels_count_2)
        labels_2, tokens = label_input(tokens, labels_2, label='##[a-zа-яйё]+|[a-zа-яйё]+', label_id=0, delit=False, labels_count=labels_count_2)
        labels_2, tokens = label_input(tokens, labels_2, label='[A-ZА-ЯЙЁ][a-zа-яйё]+', label_id=1, delit=False, labels_count=labels_count_2)
        labels_2 = check_target(labels_2, add_element_2)
        
        for index, token in enumerate(tokens):
            tokenizer_work = tokenizer(token.lower())
            tokenized_token = tokenizer_work['input_ids'][1:-1]
            lenth = len(tokenized_token)
            features.extend(tokenized_token)
            mask.extend(tokenizer_work['attention_mask'][1:-1])
            #punct
            for _ in range(lenth-1):
                target_1.append(add_element_1)
            target_1.append(labels_1[index])
            #cap
            if token.isupper():
                for _ in range(lenth):
                    target_2.append(labels_2[index])
            else:
                target_2.append(labels_2[index])
                for _ in range(lenth-1):
                    target_2.append(add_element_2)
            sub.extend(tokenizer.tokenize(token.lower()))
            
    return sub, features, mask, target_1, target_2

def label_input(tokens, features, label='.', before=None, after=None, 
                label_id=1, labels_count=4, delit=True,
                mode=None):
    compare = lambda x: re.fullmatch(label, x)
    the_last = tokens[0]
    list_to_del = []
    if len(tokens) > 2:
        the_next = tokens[2]
    else:
        the_next = None
    for index, now in enumerate(tokens):
        work = True
        if before:
            work = work and before(the_last)
        if after and the_next:
            work = work and after(the_next)
        if compare(now):
            work = work and True
        else:
            work = False
        if mode == 'punct':
            if work and not isinstance(features[index - 1], list):
                answ = [0. for i in range(labels_count)]
                answ[label_id] = 1.
                features[index - 1] = answ
                list_to_del.append(index)
        else:
            if work and isinstance(features[index], str):
                answ = [0. for i in range(labels_count)]
                answ[label_id] = 1.
                features[index] = answ
                list_to_del.append(index)
    if delit:
        for index in reversed(list_to_del):
            del features[index]
            del tokens[index]
    return features, tokens


def check_target(target, replacement):
    checked = True
    for i, j in enumerate(target):
        if not isinstance(j, list):
            checked = False
        if not checked:
            target[i] = replacement
    return target

def capitalize_after_punctuation(text):
    # Разбиваем текст на предложения по символам '.' и '?'
    sentences = re.split(r'(?<=[.?!])\s+', text)
    # Проходим по каждому предложению
    for i, sentence in enumerate(sentences):
        # Проверяем, есть ли точка или вопросительный знак в конце предложения
        if sentence[-1] in ['.', '?']:
            # Проходим по символам в предложении
            new_sentence = ''
            capitalize_next = True
            for char in sentence:
                # Если символ - буква и нужно делать заглавной
                if char.isalpha() and capitalize_next:
                    new_sentence += char.upper()
                    capitalize_next = False
                else:
                    new_sentence += char
                # Если символ - точка или вопросительный знак, следующая буква должна быть заглавной
                if char in ['.', '?']:
                    capitalize_next = True
            # Заменяем предложение в списке на обработанное
            sentences[i] = new_sentence
    # Собираем предложения обратно в текст
    result_text = ' '.join(sentences)
    return result_text

def getdata(sub, data=None, return_data=False):
    if data==None:
        tokenizer = BertTokenizer.from_pretrained(model_name) #("DeepPavlov/rubert-base-cased")
        text, features, mask, target, target2 = prepare_data(sub, tokenizer)
        input_ids = torch.tensor(features)
        attention_mask = torch.tensor(mask)
        labels1 = torch.tensor(target)
        labels2 = torch.tensor(target2)
    else:
        input_ids, attention_mask, labels1, labels2 = data
    print(input_ids.shape, attention_mask.shape, labels1.shape, labels2.shape)
    train_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels1, labels2)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512)
    if return_data:
        train_loader, [input_ids, attention_mask, labels1, labels2]
    else:
        return train_loader

def getdatalang(sub, data=None, return_data=False, lang=False, langs=None):
    if data==None:
        tokenizer = BertTokenizer.from_pretrained(model_name) #("DeepPavlov/rubert-base-cased")
        text, features, mask, target, target2, target3 = prepare_data(sub, tokenizer, lang=lang, langs=langs)
        input_ids = torch.tensor(features)
        attention_mask = torch.tensor(mask)
        labels1 = torch.tensor(target)
        labels2 = torch.tensor(target2)
        labels3 = torch.tensor(target3)
    else:
        input_ids, attention_mask, labels1, labels2, labels3 = data
    print(input_ids.shape, attention_mask.shape, labels1.shape, labels2.shape, labels3.shape)
    train_dataset = torch.utils.data.TensorDataset(input_ids, attention_mask, labels1, labels2, labels3)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=512)
    if return_data:
        train_loader, [input_ids, attention_mask, labels1, labels2]
    else:
        return train_loader

def f1_score(preds, labels, n=1):
    TP = ((preds == n) & (labels == n)).sum().float()
    FP = ((preds == n) & (labels != n)).sum().float()
    FN = ((preds != n) & (labels == n)).sum().float()
    TN = ((preds != n) & (labels != n)).sum().float()
    
    precision = TP / (TP + FP + 1e-9)
    recall = TP / (TP + FN + 1e-9)
    
    if (FN == 0 and FP == 0):
        return 1
    
    f1 = 2 * (precision * recall) / (precision + recall + 1e-9)
    
    return f1.item()


if mode=='train':
    with open(train_loader_name, 'rb') as f:
        train_loader = pickle.load(f)

with open(val_loader_name, 'rb') as f:
    val_loader = pickle.load(f)

trainer = pl.Trainer(max_epochs=max_epochs, log_every_n_steps=30)

if mode=='valid' and os.path.exists(checkpoint_way):
    checkpoint = torch.load(checkpoint_way)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
else:
    trainer.fit(model, train_loader)

res = trainer.validate(model=model, dataloaders=val_loader)
print(res)

with open(res_way, 'w') as f:
    data = json.dump({'res':res}, f, ensure_ascii=False)

torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': model.configure_optimizers().state_dict(),
            }, checkpoint_way)
