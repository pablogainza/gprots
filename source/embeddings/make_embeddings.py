import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import os
import requests
from tqdm.auto import tqdm
from Bio import SeqIO

seqs = []
for record in SeqIO.parse("cluster_P30542.fasta", "fasta"):
    seqs.append(record.seq)

seqs_bert = []
for seq in seqs: 
    seqs_bert.append(' '.join(list(seq)))

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

model = AutoModel.from_pretrained("Rostlab/prot_bert")

fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0)

#sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

print('Starting embedding')
embedding = fe(seqs_bert)

features = [] 

for seq_num in range(len(embedding)):
    seq_len = len(seqs_bert[seq_num].replace(" ", ""))
    start_Idx = 1
    end_Idx = seq_len+1
    seq_emd = embedding[seq_num][start_Idx:end_Idx]
    features.append(seq_emd)

print(len(list(seqs[0])))
print(len(features[0]))

