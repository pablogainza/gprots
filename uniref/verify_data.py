import os 
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
            if line2.rstrip() in all_acc: 
                print(f'\t{uniref_acc_to_gene[line2.rstrip()]}')
        print(' ')

