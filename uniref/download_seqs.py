import os 
import wget 
gene_to_uniref_acc = {}
uniref_acc_to_gene = {}
all_acc = set()
with open ('mappings.csv') as f: 
    for line in f.readlines(): 
        fields = line.split(',')
        genename = fields[2].rstrip()
        acc = fields[0]
        gene_to_uniref_acc[genename] = acc
        uniref_acc_to_gene[acc] = genename
        all_acc.add(acc)

for gene in gene_to_uniref_acc.keys():
    uniref_fn = f'{gene}.uniref'
    with open(os.path.join(gene, uniref_fn)) as f2: 
        print(f'{gene}')
        for line2 in f2.readlines():
            acc = line2.rstrip()
            if gene == 'DRD1':
                result = wget.download(f"https://www.uniprot.org/uniprot/?query=accession:{acc}&format=fasta", \
                            os.path.join(gene, f'cluster_{acc}.fasta'))
            else:
                continue
#            elif not os.path.exists(os.path.join(gene, f'cluster_{acc}.fasta')):
#                result = wget.download(f"https://www.uniprot.org/uniprot/?query=cluster:(uniprot:{acc}%20identity:0.5)&format=fasta", \
#                            os.path.join(gene, f'cluster_{acc}.fasta'))
#                print(result)
#                break
        print(' ')

