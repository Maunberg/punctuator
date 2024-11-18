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
from IPython.display import FileLink
import random
import argparse
random.seed(42)

from datasets import Dataset
import os; import psutil; import timeit
from datasets import load_dataset
import datasets

model_name = 'DeepPavlov/rubert-base-cased' #google-bert/bert-base-multilingual-cased
way_to_corpus = ['data_raw/tedtalks_en.txt',
                 'data_raw/books_en.txt']
n = 5_000_000
lang = 'en'
version = 2
lenth_train = 450_000
lenth_valid = 20_000
way_to_norm_text = 'data_norm/en.txt'

train_loader_name = f'data_load/{lang}_train_loader_v{version}_{lenth_train}.pkl'
val_loader_name = f'data_load/{lang}_val_loader_v{version}_{lenth_valid}.pkl'


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


def normalize_line(line: str):
    line = re.sub('http[^ ]+ ', '', line)
    line = re.sub('#.+#:', ' ', line)
    line = re.sub('!;', '.', line)
    #line = re.sub(':', ',', line)
    line = re.sub('\xa0', ' ', line)
    line = re.sub(' - ', ' – ', line)
    #line = re.sub('([^A-Za-zА-Яа-яйёЙЁ ]-[^A-Za-zА-Яа-яйёЙЁ ])', ' ', line)
    line = re.sub('^[-–] ', ' ', line)
    line = re.sub('[^A-Za-zА-Яа-я0-9йЙёЁäöüßÄÖÜ,\.\?–\-:]', ' ', line)
    line = re.sub('([\.,\?:–])[\.,\?:– ]+', r' \1 ', line)
    line = re.sub('([\.,\?:–])', r' \1 ', line)
    line = re.sub('([\.,?:]) +[-–] ', r'\1 ', line)
    line = line.strip()
    
#     if line[-1] not in ['.', '?']:
#         line+=' PBR'
#     elif line[-1]== '.':
#         line = line[:-1]+' PBR'
#     elif line[-1]== '?':
#         line = line[:-1]+' QBR'
    
#     line = line.replace('.\n', 'PBR')
#     line = line.replace('?\n', 'QBR')
    line = re.sub('\n', ' ', line)
    line = re.sub(' +', ' ', line)
    return capitalize_after_punctuation(line.strip())

def normalize_distr(subf: list):
    sub = ' '.join(subf).replace('\n', '')
    sub = re.sub('([.?])', r'\1SEP', sub)
    sub = re.split(r'SEP', sub)
    sub = [i for i in sub if len(i)>2]
    sub_renew = []
    now_lenth = 0
    sen = sub[0]
    for i in tqdm(sub):
        now_lenth = len([word for word in sen.split() if len(word)>1])
        if now_lenth>55:
            sen = re.sub(' +', ' ', sen)
            sub_renew.append(sen)
            sen = ''
        else:
            sen += ' ' + i
    print(f'was: {len(subf):8} distr = {sum([len([word for word in sen_was.split() if len(word)>1]) for sen_was in subf])/len(subf)}')
    print(f'now: {len(sub_renew):8} distr = {sum([len([word for word in sen_new.split() if len(word)>1]) for sen_new in sub_renew])/len(sub_renew)}')
    return sub_renew


def normalize_text(lines):
    sub = []
    for line in tqdm(lines):
        if '.' not in line and '?' not in line:
            continue
        line = normalize_line(line)
        if len(line)>2:
            sub.append(line)
    return sub

if not os.path.exists(way_to_norm_text):
    if len(way_to_corpus) == 1:
        with open(way_to_corpus[0], 'r', encoding="utf8") as file:
            lines = file.readlines()[:n]
    else:
        lines = []
        for way_crp in way_to_corpus:
            with open(way_crp, 'r', encoding="utf8") as file:
                lines.extend(file.readlines())
        lines = lines[:n]

    sub = normalize_text(lines)
    sub = normalize_distr(sub)

    with open(way_to_norm_text, "w") as file:
        text_to_write = '\n'.join(sub)
        file.write(text_to_write)
else:
    with open(way_to_norm_text, 'r', encoding="utf8") as file:
        sub = file.readlines()[:n]

print('FULL LENGHT = ', len(sub))

if not os.path.exists(train_loader_name):
    train_loader = getdata(sub[:lenth_train])
    with open(train_loader_name, 'wb') as f:
        pickle.dump(train_loader, f)

if len(sub) < (lenth_train+lenth_valid):
    lenth_valid = len(sub) - lenth_train

if not os.path.exists(val_loader_name):
    val_loader = getdata(sub[-lenth_valid:])

    with open(val_loader_name, 'wb') as f:
        pickle.dump(train_loader, f)
