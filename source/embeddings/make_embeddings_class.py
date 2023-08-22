import torch
from transformers import AutoTokenizer, AutoModel, pipeline
import re
import numpy as np
import os
import requests
from tqdm.auto import tqdm
from Bio import SeqIO
import sys
# Make embeddings of sequences using prottrans
class Embeddings: 
    def __init__(self):
        tokenizer = AutoTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)

        model = AutoModel.from_pretrained("Rostlab/prot_bert")

        self.fe = pipeline('feature-extraction', model=model, tokenizer=tokenizer,device=0)

    def do_embeddings(self, seqs):
        seqs_bert = [' '.join(list(seq)) for seq in seqs]
        embedding = self.fe(seqs_bert)
        features = []

        for seq_num in range(len(embedding)):
            seq_len = len(seqs_bert[seq_num].replace(" ", ""))
            start_Idx = 1
            end_Idx = seq_len+1
            seq_emd = embedding[seq_num][start_Idx:end_Idx]
            features.append(seq_emd)
        assert(len(features) == len(seqs))
        return np.array(features)


if __name__ == "__main__":
    embed = Embeddings()
    embed.do_embeddings(\
            ['MNGTEGPNFYVPFSNATGVVRSPFEYPQYYLAEPWQSMLAAYMFLLIVLGFPINFLTLYVTVQHKKLRTPLNYILLNLAVADLFMVLGGFTSTLYTSLHGYFVFGPTGCNLEGFFATLGGEIALWSLVVLAIERYVVVCKPMSNFRFGENHAIMGVAFTWVMALACAAPPLAGWSRYIPEGLQCSCGIDYYTLKPEVNNESFVIYMFVVHFTIPMIIIFFCYGQLVFTVKEAAAQQQESATTQKAEKEVTRMVIIMVIAFLICWVPYASVAFYIFTHQGSNFGPIFMTIPAFFAKSAAIYNPVIYIMMNKQFRNCMLTTICCGKNPLGDDEASATVSKTETSQVAPA']
            )


