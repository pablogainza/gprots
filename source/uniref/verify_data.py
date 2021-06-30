import os 
import pandas as pd
from Bio import SeqIO
gene_to_uniref_acc = {}
uniref_acc_to_gene = {}
all_human_acc = set()
all_acc_by_cluster = {}
df = pd.read_csv('../ground_truth.csv')
acc_to_gene = {}

for ix, row in df.iterrows():
    name = row['HGNC']
    print(name)
    acc = row['UniprotAcc']
    uname = row['UniprotName']
    gene_to_uniref_acc[name] = acc
    uniref_acc_to_gene[acc] = name
    all_human_acc.add(acc)
    in_human = os.path.join(os.path.join(name, f'{acc}.fasta'))
    in_cluster = os.path.join(name, f'cluster_{acc}.fasta')
    acc_to_gene[acc] = name
    if not (os.path.exists(in_human)): 
        print(f"Fasta doesnt exist for {name} {acc}")
        continue
    if not (os.path.exists(in_cluster)): 
        print(f"Cluster doesnt exist for {name} {acc}")
        continue
    # Open cluster_fasta
    num_records_human = 0
    for record in SeqIO.parse(in_human, "fasta"):
        num_records_human += 1
    num_records_cluster = 0
    all_acc_by_cluster[acc] = []
    for record in SeqIO.parse(in_cluster, "fasta"):
        all_acc_by_cluster[acc].append(record.id.split('|')[1])
        num_records_cluster+= 1
    print(f"{name} {acc} num_records_human_seq: {num_records_human}, num_records_cluster: {num_records_cluster}")

errors = ''
for key in all_acc_by_cluster:
    if key not in all_acc_by_cluster[key]:
        errors += f'error, {acc_to_gene[key]} ({key}) is not in its own cluster!\n'
    for acc in all_human_acc: 
        if acc in all_acc_by_cluster[key]:
            print(f'{acc} in the cluster of {acc}')
    print(' ')
    
print(errors)
