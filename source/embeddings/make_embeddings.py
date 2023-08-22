import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import os
import requests
from tqdm.auto import tqdm
from Bio import SeqIO
import sys

in_fasta = sys.argv[1]
outdir = sys.argv[2]
if not os.path.exists(outdir):
    os.makedirs(outdir)

seqs = []
acc = []
for record in SeqIO.parse(in_fasta, "fasta"):
    acc.append(record.id.split('|')[1])
    seqs.append(record.seq)

seqs_bert = []
for seq in seqs: 
    seqs_bert.append(' '.join(list(seq)))

tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

model = AutoModel.from_pretrained("Rostlab/prot_bert")

fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0)

#sequences_Example = [re.sub(r"[UZOB]", "X", sequence) for sequence in sequences_Example]

print('Starting embedding: ', in_fasta, outdir)
# split into files of size 100
splits = np.arange(0,len(seqs_bert), 100)
splits = np.append(splits, len(seqs_bert))
split_embeddings = []
for i in range(len(splits)-1):
    embedding = fe(seqs_bert[splits[i]:splits[i+1]])
    split_embeddings.extend(embedding)

embedding = split_embeddings

features = [] 

for seq_num in range(len(embedding)):
    seq_len = len(seqs_bert[seq_num].replace(" ", ""))
    start_Idx = 1
    end_Idx = seq_len+1
    seq_emd = embedding[seq_num][start_Idx:end_Idx]
    features.append(seq_emd)

assert(len(features) == len(seqs))
for ix, uniprotid in enumerate(acc): 
    outfn_seq = os.path.join(outdir, uniprotid+'_seq.npy')
    np.save(outfn_seq, seqs[ix])
    
    outfn_feat = os.path.join(outdir, uniprotid+'_feat.npy')
    np.save(outfn_feat, features[ix])
    assert(len(features[ix]) == len(list(seqs[ix])))


