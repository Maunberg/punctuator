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

class model(pl.LightningModule):
    def __init__(self, num_classes, pretrained_model_name="bert-base-uncased", freeze_bert=False):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.fc1 = nn.Linear(768, num_classes)
        self.relu = nn.ReLU()
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)
        logits1 = self.fc1(last_hidden_state)
        return logits1

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels1, _ = batch
        input_ids =  torch.tensor([[i for i in input_ids]]).cuda()
        attention_mask =  torch.tensor([[i for i in attention_mask]]).cuda()
        labels1 = labels1.cuda()
        logits1 = self(input_ids, attention_mask)
        logits1 = logits1.squeeze()
        loss1 = nn.CrossEntropyLoss()(logits1, labels1)
        
        preds = torch.argmax(logits1, dim=-1).view(-1)
        labels = torch.argmax(labels1, dim=-1).view(-1)
        f1_sp = f1_score(preds, labels, n=0)
        f1_fs = f1_score(preds, labels, n=1) 
        f1_cm = f1_score(preds, labels, n=2)
        f1_qe = f1_score(preds, labels, n=3)
        self.log('loss1', loss1, prog_bar=True)
        self.log('f1 _', f1_sp, prog_bar=True)
        self.log('f1 .', f1_fs, prog_bar=True)
        self.log('f1 ,', f1_cm, prog_bar=True)
        self.log('f1 ?', f1_qe, prog_bar=True)
        #self.log('f1 br', f1_br, prog_bar=True)
        #print('res', check_result())
        
        return loss1

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels1, _ = batch
        input_ids =  torch.tensor([[i for i in input_ids]]).cuda()
        attention_mask =  torch.tensor([[i for i in attention_mask]]).cuda()
        labels1 = labels1.cuda()
        logits1 = self(input_ids, attention_mask)
        logits1 = logits1.squeeze()
        
        preds = torch.argmax(logits1, dim=-1).view(-1)
        labels = torch.argmax(labels1, dim=-1).view(-1)
        f1_sp = f1_score(preds, labels, n=0)
        f1_fs = f1_score(preds, labels, n=1) 
        f1_cm = f1_score(preds, labels, n=2)
        f1_qe = f1_score(preds, labels, n=3)
        self.log('f1 _', f1_sp, prog_bar=True)
        self.log('f1 .', f1_fs, prog_bar=True)
        self.log('f1 ,', f1_cm, prog_bar=True)
        self.log('f1 ?', f1_qe, prog_bar=True)
        return {'f1 _':f1_sp, 'f1 .': f1_fs, 'f1 ,':f1_cm, 'f1 ?':f1_qe}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=2e-5)
        return optimizer
