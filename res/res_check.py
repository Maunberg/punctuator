import os
import json

lang = 'ende'
version = 'v3'

files = [i for i in os.listdir() if i.count('BERT') and i.count(lang) and i.count(version)]

result_pun = {}
result_cap = {}
for i in files:
    with open(i) as f:
        data = json.load(f)['res'][0]
    if 'f1 :' in data:
        result_pun[i.split('_')[0]] = (data['f1 .'] + data['f1 ,']\
                                      + data['f1 ?'] + data['f1 :'] + data['f1 -'])/5
    else:
        result_pun[i.split('_')[0]] = (data['f1 .'] + data['f1 ,'] + data['f1 ?'])/3
    result_cap[i.split('_')[0]] = (data['f1 ABBR'] + data['f1 Upper'])/2

print('PUNCTUATION RATE')
result_pun = dict(sorted(result_pun.items(), key=lambda x: x[1]))
for i in result_pun:
    print(i, result_pun[i])

print('\n\nCAPITALIZATION RATE')
result_cap = dict(sorted(result_cap.items(), key=lambda x: x[1]))
for i in result_cap:
    print(i, result_cap[i])
