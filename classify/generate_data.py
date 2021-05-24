# Read the ground truth.
import os
from subprocess import Popen, PIPE
import numpy as np
from IPython.core.debugger import set_trace
import pandas as pd 
gtdf = pd.read_csv('../ground_truth.csv')

# Read the PDB of the template structure - for now 6OIJ.
from Bio.PDB import *
from Bio.SeqUtils import seq1
parser = PDBParser()
struct = parser.get_structure('6OIJ.pdb', '6OIJ.pdb')

# Compute the sequence identity between rseq and qseq, after alignment
def seq_id(rseq, qseq):
    n = len(list(rseq))
    identical = 0
    for i in range(n):
        if rseq[i] == qseq[i]:
            identical += 1
    return identical/n

# Align qseq to rseq with clustal, return the residues in qseq that correspond to the interface residues in rseq.
def align_seqs(rseq, qseq, rinterface=None):
    # Save rseq and qseq to temp.fasta
    with open ('temp.fasta', 'w+') as f:
        f.write('>refseq\n')
        f.write(''.join(rseq)+'\n')
        f.write('>queryseq\n')
        f.write(''.join(qseq)+'\n')

    # Invoke clustal
    args = ['utils/clustal', \
            '-i', \
            'temp.fasta']

    p2 = Popen(args, stdout=PIPE, stderr=PIPE)
    stdout, stderr = p2.communicate()

    # Read the alignment
    stdout = stdout.decode("utf-8")
    lines = stdout.split('\n')
    i = 1 
    raln = ''
    qaln = ''
    while not lines[i].startswith('>'):
        raln += lines[i]
        i += 1
    i = i+1 
    while i < len(lines):
        qaln += lines[i]
        i += 1
        

    # Compute the sequence identity between the sequences, in two directions.
    sequence_identity = max(seq_id(raln, qaln), seq_id(qaln, raln))
    print(raln)
    print(qaln)

    return sequence_identity

# Compute the interface residues, size N, of the template structure and label them. 
all_chains = Selection.unfold_entities(struct, 'C')
receptor_atoms = None
g_atoms = []
for chain in all_chains:
    if chain.id == 'R':
        receptor_atoms = Selection.unfold_entities(chain, 'A')
        receptor_res = Selection.unfold_entities(chain, 'R')
    else:
        g_atoms.extend(Selection.unfold_entities(chain, 'A'))

from scipy.spatial import cKDTree
rec_coord = [x.get_coord() for x in receptor_atoms]
g_coord = [x.get_coord() for x in g_atoms]
kdt = cKDTree(g_coord)
d, r = kdt.query(rec_coord)
interface_atoms = np.where(d < 10.0)[0]
interface_res = [receptor_atoms[x].get_parent().get_id() for x in interface_atoms]
interface_res = set(interface_res)
# print for debug in pymol 
#outline = ''
#for x in interface_res:
#    outline += f'{x[1]}+'
#print(outline)

# Assign a flag to the interface residues.
template_seq = []
template_iface_flags = []
for res in receptor_res: 
    template_seq.append(seq1(res.resname))
    if res.id in interface_res:
        template_iface_flags.append(1)
    else:
        template_iface_flags.append(0)

#print(''.join(template_seq))

# For each GPCR in our dataset 
dataset_dir = '../uniref/{}/in_data/'

all_gpcrs = gtdf['GeneName'].tolist()
human_uniprotid = gtdf['UniprotAcc'].tolist()
# Go through every GPCR 
for ix, gpcr in enumerate(all_gpcrs):
    curdir = dataset_dir.format(gpcr)
    # but first read the human sequence. 
    human_seq = np.load(os.path.join(curdir, human_uniprotid[ix]+'_seq.npy'))
    identity_to_templ = align_seqs(template_seq, human_seq, rinterface=None)
    print(f'Seq id to template = {identity_to_templ}')
    #print(''.join(human_seq))
    # Now go through each sequence in UNIREF50 for the GPCR
    for fn in os.listdir(curdir):
        if fn.endswith('seq.npy'):
            acc = fn.split('_')[0]
            seq = np.load(os.path.join(curdir, fn))
#            identity_to_human = align_seqs(human_seq, seq, rinterface=None)
#            print(f'Seq id to human = {identity_to_human}')
#            identity_to_templ = align_seqs(template_seq, seq, rinterface=None)
#            print(f'Seq id to template = {identity_to_templ}')

    break



        # Align to template sequence (use clustal?)  

        # for the N residues in the interface get the 'fingerprints'

        # Save a tensor of size  (N,1024) containing the input data for this training instance. 

        # If the UNIREF sequence is the corresponding one for human, save it as HUMAN.npy

        # Save the ground truth in numpy for easy access.
