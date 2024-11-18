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
from model_list.BERT_v3 import model as full_model

random.seed(42)

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


class model(full_model):
    def __init__(self, pretrained_model_name="DeepPavlov/rubert-base-cased", 
                 freeze_bert=False, rnn_hidden_dim=128, num_rnn_layers=1, 
                 bidirectional=False, num_classes1=6, num_classes2=3, dropout=0.1):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        
        self.rnn = nn.RNN(768, rnn_hidden_dim, num_layers=num_rnn_layers, bidirectional=bidirectional, batch_first=True)
        if bidirectional:
            rnn_out_dim = rnn_hidden_dim * 2
        else:
            rnn_out_dim = rnn_hidden_dim
        
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(rnn_out_dim, num_classes1)
        self.fc2 = nn.Linear(rnn_out_dim, num_classes2)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        rnn_outputs, sd = self.rnn(last_hidden_state)
        rnn_outputs = self.dropout(rnn_outputs)
        logits1 = self.fc1(rnn_outputs)
        logits2 = self.fc2(rnn_outputs)
        return logits1, logits2
