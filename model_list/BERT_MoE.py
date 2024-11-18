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

class ExpertMoE(nn.Module):
    def __init__(self, input_size, num_experts, output_size, expert_size=768):
        super(ExpertMoE, self).__init__()
        self.num_experts = num_experts
        self.experts = nn.ModuleList([nn.Linear(input_size, expert_size) for _ in range(num_experts)])
        self.gate = nn.Linear(input_size, num_experts)
        self.output = nn.Linear(expert_size, output_size)

    def forward(self, input):
        expert_outputs = [expert(input) for expert in self.experts]
        gate_values = torch.softmax(self.gate(input), dim=-1)
        aggregated_output = sum(expert_output * gate_value.unsqueeze(-1) for expert_output, gate_value in zip(expert_outputs, gate_values.unbind(dim=-1)))
        output = self.output(aggregated_output)
        return output

class model(full_model):
    def __init__(self, pretrained_model_name, freeze_bert=False, num_experts=2):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.moe1 = ExpertMoE(input_size=768, num_experts=num_experts, output_size=6)
        self.moe2 = ExpertMoE(input_size=768, num_experts=num_experts, output_size=3)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=-1)
        
    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        last_hidden_state = self.dropout(last_hidden_state)
        logits1 = self.moe1(last_hidden_state)
        logits2 = self.moe2(last_hidden_state)
        return logits1, logits2

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=2e-6)
        return optimizer 


