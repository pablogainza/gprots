#!//usr/local/opt/python@3.9/bin/python3.9
import sys
sys.path.append('/usr/local/Cellar/pymol/2.4.0_3/libexec/lib')
sys.path.append('/usr/local/Cellar/pymol/2.4.0_3/libexec/lib/python3.9/site-packages')
# First line should point to pymol path.  
from pymol import cmd, stored

import os
from subprocess import Popen, PIPE
import numpy as np
from IPython.core.debugger import set_trace
import pandas as pd 


# Read the ground truth.
gtdf = pd.read_csv('../ground_truth.csv')

# Read the PDB of the template structure - for now 7jvq.
cmd.load('7jvq.pdb', 'template')
cmd.select('Rchain', 'template and chain R')
cmd.create('templateR', 'Rchain') 

# one_letter["SER"] will now return "S"
one_letter ={'VAL':'V', 'ILE':'I', 'LEU':'L', 'GLU':'E', 'GLN':'Q', \
        'ASP':'D', 'ASN':'N', 'HIS':'H', 'TRP':'W', 'PHE':'F', 'TYR':'Y',    \
        'ARG':'R', 'LYS':'K', 'SER':'S', 'THR':'T', 'MET':'M', 'ALA':'A',    \
        'GLY':'G', 'PRO':'P', 'CYS':'C', 'CLR': '', 'PLM': ''}


def getIfaceInQuery(align_fn, gpcr_name, iface_ix): 
    """
       Parse align_fn in clustal format and get the alignment between template 
       and query. Then map the iface_ix onto the query sequence. 

       Returns: for iface_ix in the query (gpcr_name)
    """
    template_aln_seq = []
    query_aln_seq = []
    with open(align_fn) as f: 
        for line in f.readlines():
            if line.startswith('templateR'): 
                entry = list(line.rstrip())
                template_aln_seq.extend(entry[13:])
            elif line.startswith(gpcr_name):
                entry = list(line.rstrip())
                query_aln_seq.extend(entry[13:])
        
    # Now, go through the template_aln and select the interface vertices, ignore dashes. 
    count_not_dash = -1
    iface_in_aln = []
    for i in range(len(template_aln_seq)):
        if template_aln_seq[i] != '-':
            count_not_dash += 1
            if count_not_dash in iface_ix:
                iface_in_aln.append(i)
    
    # Go through query_aln and selct the interface vertices based on iface_in_aln, ignoring dashes as well.
    count_not_dash = -1
    iface_in_query = []
    for i in range(len(query_aln_seq)):
        if query_aln_seq[i] != '-':
            count_not_dash+= 1
            if i in iface_in_aln:
                iface_in_query.append(count_not_dash)

    return iface_in_query    



# Compute the sequence identity between rseq and qseq, after alignment
def seq_id(rseq, qseq):
    n = len(list(rseq))
    identical = 0
    for i in range(n):
        if rseq[i] == qseq[i]:
            identical += 1
    return identical/n

# Align qseq to rseq with clustal, return the residues in qseq that correspond to the interface residues in rseq.
def align_seqs(rseq, qseq, rinterface):
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
    
    # Compute the mapping of interface in rseq to qseq. 

    # go through the reference raln and select the interface vertices, ignore dashes. 
    count_not_dash = -1
    iface_in_aln = []
    for i in range(len(raln)):
        if raln[i] != '-':
            count_not_dash += 1
            if count_not_dash in rinterface:
                iface_in_aln.append(i)
    
    # Go through query_aln and select the interface vertices based on iface_in_aln, ignoring dashes as well.
    count_not_dash = -1
    iface_in_query = []
    identical_iface = 0.0
    total_iface = 0.0
    for i in range(len(qaln)):
        if qaln[i] != '-':
            count_not_dash += 1
            if i in iface_in_aln:
                iface_in_query.append(count_not_dash)
                total_iface += 1
                if raln[i] == qaln[i]:
                    identical_iface += 1
   
    print('Fraction identical in interface: {:.3f}, total: {}'.format(identical_iface/total_iface, total_iface))

    # Compute the sequence identity between the sequences, in two directions.
    sequence_identity = max(seq_id(raln, qaln), seq_id(qaln, raln))
    print(raln)
    print(qaln)

    return sequence_identity

# Compute the interface residues, size N, of the template structure and label them. 
cmd.select('iface_res', 'template and byRes((chain A or chain B) around 10)')
cmd.select('ifaceR', f'iface_res and chain R')
stored.iface = []
cmd.iterate('ifaceR and name ca', 'stored.iface.append(resi)')

# Compute the sequence of the template.
stored.template_seq = []
cmd.iterate('name ca and templateR', 'stored.template_seq.append(resn)')
template_seq = [one_letter[stored.template_seq[x]] for x in range(len(stored.template_seq))]

# and the residue id of each one.
stored.template_resi = []
cmd.iterate('name ca and templateR', 'stored.template_resi.append(resi)')
template_resi = stored.template_resi

assert(len(stored.template_resi) == len(stored.template_seq))

# Iface indices in template sequence. 
iface_ix = []
for i in range(len(stored.iface)):
    iface_ix.append(stored.template_resi.index(stored.iface[i]))

print('Total: {} residues in interface'.format(len(iface_ix)))
iface_seq_templ = [template_seq[x] for x in iface_ix]

# Pretrained data directory:
dataset_dir = '../uniref/{}/in_data/'

all_gpcrs = gtdf['GeneName'].tolist()
human_uniprotid = gtdf['UniprotAcc'].tolist()
pdb_names= gtdf['PDB'].tolist()

# For each GPCR in our dataset 
for ix, gpcr in enumerate(all_gpcrs):
    # Read the PDB of the corresponding human GPCR
    cmd.load(f'../data01_gpcrs/{pdb_names[ix]}', gpcr)

    # Compute the sequence of the template.
    stored.query_seq = []
    cmd.iterate(f'name ca and {gpcr}', 'stored.query_seq.append(resn)')
    query_seq = [one_letter[stored.query_seq[x]] for x in range(len(stored.query_seq))]

    # Use Pymol to align the human GPCR to the template. 
    align_ret = cmd.align(gpcr, 'templateR', object=f'aln_{gpcr}')

    # Assert that the alignment RMSD < 10.0
    print(f"RMSD: {align_ret[0]}, atoms after: {align_ret[1]}, atoms before: {align_ret[4]}")
    

    # Mark the interface residues in the aligned PDB to correspond exactly to those in the template. 
    # It is very important here that if there are gaps in this GPCR (i.e. interface residues not present)
    # Then the gaps are preserved. For this we preserve the dash.
    cmd.save(f'{gpcr}.aln',f'aln_{gpcr}')

    # We now need a mapping from the residues in the interface of the template to those in the 
    # new GPCR. 
    iface_in_query = getIfaceInQuery(f'{gpcr}.aln', gpcr, iface_ix)

    # and the residue id of each one.
    stored.query_resi = []
    cmd.iterate(f'name ca and {gpcr}', 'stored.query_resi.append(resi)')
    query_resi = stored.query_resi
    query_abs_resi =[]
    outdebug = ''
    for jj, res in enumerate(query_resi):
        if jj in iface_in_query:
            query_abs_resi.append(res)

    
    # Load the human sequence and align to the 'query' sequence, which is the PDB model from GPCRDB.
    curdir = dataset_dir.format(gpcr)
    human_seq = np.load(os.path.join(curdir, human_uniprotid[ix]+'_seq.npy'))

    identity_to_templ = align_seqs(query_seq, human_seq, rinterface=iface_in_query)

#    print(f'Seq id to template = {identity_to_templ}')
    #print(''.join(human_seq))
    # Now go through each sequence in UNIREF50 for the GPCR and align it to the prealigned human model. 
#    for fn in os.listdir(curdir):
#        if fn.endswith('seq.npy'):
#            acc = fn.split('_')[0]
#            seq = np.load(os.path.join(curdir, fn))
            # For sanity, check that the sequence has X identity. Also store the identity. 
            # Perform the actual alignment to the template and check the identity of the sequence.
#            identity_to_templ = align_seqs(template_seq, seq, rinterface=None)
#            print(f'Seq id to template = {identity_to_templ}')




        # Align to template sequence (use clustal?)  

        # for the N residues in the interface get the 'fingerprints'

        # Save a tensor of size  (N,1024) containing the input data for this training instance. 

        # If the UNIREF sequence is the corresponding one for human, save it as HUMAN.npy

        # Save the ground truth in numpy for easy access.
