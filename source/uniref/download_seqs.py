import os 
import wget 
import pandas as pd
gene_to_uniref_acc = {}
uniref_acc_to_gene = {}
all_acc = set()
df = pd.read_csv('../ground_truth.csv')
for ix, row in df.iterrows():
    name = row['HGNC']
    print(name)
    acc = row['UniprotAcc']
    uname = row['UniprotName']
    gene_to_uniref_acc[name] = acc
    uniref_acc_to_gene[acc] = name
    all_acc.add(acc)
    if not os.path.exists(name):
        os.makedirs(name)
    # Download sequence of human protein. 
    out_human = os.path.join(os.path.join(name, f'{acc}.fasta'))
    if not os.path.exists(out_human):
        result = wget.download(f"https://www.uniprot.org/uniprot/?query=accession:{acc}&format=fasta", \
               os.path.join(name, f'{acc}.fasta'))
    # Download cluster. 
    out_cluster = os.path.join(name, f'cluster_{acc}.fasta')
    if not os.path.exists(out_cluster):
        result = wget.download(f"https://www.uniprot.org/uniprot/?query=cluster:(uniprot:{acc}%20identity:0.5)&format=fasta", \
            out_cluster)
        print(f'Result = {result}')

