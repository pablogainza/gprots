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

# set the iface cutoff
iface_cutoff = 8

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
       and query. Then map the interface residues of the template (defined by index in iface_ix)
       onto the query sequence. 

       Returns: the aligned query and the residues corresponding to iface_ix \
           in the ALIGNED query defined in (gpcr_name)
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
        
    # Now, go through the template_aln and select the interface vertices.
    # For the template it is very important to ignore the dashes. 
    count_not_dash = -1
    iface_in_aln = []
    for i in range(len(template_aln_seq)):
        if template_aln_seq[i] != '-':
            count_not_dash += 1
            if count_not_dash in iface_ix:
                iface_in_aln.append(i)


    return query_aln_seq, iface_in_aln
    
def assign_ix_to_aln(aln):
    """ 
    Assign a number to every residue in aln according to its own indexing. 
    use 'insertion' to account for gaps.
    """
    aln_ix = []
    cur_aln = -1
    cur_ins = 0
    for i in range(len(aln)):
        if aln[i] == '-':
            cur_ins += 1
        else: 
            cur_ins = 0
            cur_aln += 1
        aln_ix.append((cur_aln, cur_ins))
    return aln_ix


def seq_id(rseq, qseq):
    """
        Compute the sequence identity between rseq and qseq, after alignment, ignoring dashes.
    """
    if len(rseq) == 0 or len(qseq) == 0:
        return 0
    n = 0.0
    identical = 0
    for i in range(len(rseq)):
        if rseq[i] != '-' and qseq[i] != '-':
            n+= 1.0
            if rseq[i] == qseq[i]:
                identical += 1.0
    return identical/n

def align_seqs(rseq, qseq, rinterface):
    """
    Align qseq to rseq with clustal, return the residues in qseq that correspond to the interface residues in rseq.

        rseq: reference sequence (potentially with gaps '-')
        qseq: query sequence. 
        rinterface: indices in rseq (including gaps) that are interface.

    Returns: 
        iface_in_query: interface residues in the query sequence, in order. 
            if a residue is missing the query, then a -1 is set at that position.

    """
    
    # rseq has gaps with respect to the template. Keeping track of these gaps is important
    # to select always the same residues. To achieve this, we must remove them first and add them later
    rseq_gapless = []
    count_start_gaps = 0
    for ix, aa in enumerate(list(rseq)):
        if aa == '-':
            rseq_gapless.append('x')
            count_start_gaps += 1
        else:
            rseq_gapless.append(aa)

    # Save rseq_gapless and qseq to temp.fasta
    with open ('temp.fasta', 'w+') as f:
        f.write('>refseq\n')
        f.write(''.join(rseq_gapless)+'\n')
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
    i = 0
    raln = ''
    qaln = ''
    while not lines[i].startswith('>refseq'):
        i += 1
    i = i+1
    while not lines[i].startswith('>'):
        raln += lines[i]
        i += 1
    i = i+1 
    while i < len(lines):
        qaln += lines[i]
        i += 1

    # Assign a number to every residue in raln and qaln according to its own indexing. 
    # use 'insertion' to account for gaps.
    raln_ix = assign_ix_to_aln(raln)
    qaln_ix = assign_ix_to_aln(qaln)
    
    # The last insertion must correspond to the length.
    assert (raln_ix[-1][0] == len(rseq)-1)
    assert (qaln_ix[-1][0] == len(qseq)-1)

    # Now get the interface residues in raln based on qaln.
    iface_in_query = []
    qiface_seq = []
    identical_iface = 0.0
    for i in range(len(raln_ix)):
        if raln_ix[i][0] in rinterface and raln_ix[i][1] == 0:
            if qaln[i] == '-':
                iface_in_query.append(-1)
                qiface_seq.append('-')
            else:
                assert(qaln_ix[i][1] == 0)
                iface_in_query.append(qaln_ix[i][0])
                qiface_seq.append(qaln[i])
            if raln[i] == qaln[i]:
                identical_iface += 1

    print(''.join(qiface_seq))
    print('Fraction identical in interface: {:.3f}, total: {}'.format(identical_iface/len(rinterface), identical_iface))

    sequence_identity = max(seq_id(raln, qaln), seq_id(qaln, raln))
    print(f"Sequence identity: {seq_id(raln,qaln)}, {seq_id(qaln, raln)}")

    return iface_in_query, sequence_identity, qiface_seq

# Compute the interface residues, size N, of the template structure and label them. 
cmd.select('iface_res', f'template and byRes((chain A or chain B) around {iface_cutoff})')
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
    if gpcr == 'PTGDR':
        print("Ignoring PTGDR for now") 
        continue

    # Read the PDB of the corresponding human GPCR
    cmd.load(f'../data01_gpcrs/{pdb_names[ix]}', gpcr)

    # Compute the sequence of the template.
    stored.query_seq = []
    cmd.iterate(f'name ca and {gpcr}', 'stored.query_seq.append(resn)')
    query_seq = [one_letter[stored.query_seq[x]] for x in range(len(stored.query_seq))]

    # Use Pymol to align the human GPCR to the template. 
    align_ret = cmd.align(gpcr, 'templateR', object=f'aln_{gpcr}')

    # Assert that the alignment RMSD < 10.0
    print(f"{gpcr} RMSD: {align_ret[0]}, atoms after: {align_ret[1]}, atoms before: {align_ret[4]}")
    

    # Mark the interface residues in the aligned PDB to correspond exactly to those in the template. 
    # It is very important here that if there are gaps in this GPCR (i.e. interface residues not present)
    # Then the gaps are preserved. For this we preserve the dash.
    cmd.save(f'/tmp/{gpcr}.aln',f'aln_{gpcr}')

    # We now need a mapping from the residues in the interface of the template to those in the 
    # new GPCR. 
    query_aln, iface_in_query_aln = getIfaceInQuery(f'/tmp/{gpcr}.aln', gpcr, iface_ix)

    # Print how many are identical to the template.
    print(''.join(np.asarray(template_seq)[iface_ix]))
    print(''.join(np.asarray(query_aln)[iface_in_query_aln]))
    count_identical = 0.0
    for i in range(len(iface_ix)):
        if np.asarray(template_seq)[iface_ix][i] == np.asarray(query_aln)[iface_in_query_aln][i]:
            count_identical+=1 
    print('count_identical: {:.2f}'.format(count_identical/len(iface_ix)))

    # Load the human sequence and align to the 'query' sequence, which is the PDB model from GPCRDB.
    curdir = dataset_dir.format(gpcr)
    human_seq = np.load(os.path.join(curdir, human_uniprotid[ix]+'_seq.npy'))

    iface_in_human_seq, _, _= align_seqs(query_aln, human_seq, rinterface=iface_in_query_aln)

    # Now, as a verification of the alignments, color the residues in the interface in 
    # red using the numbering of the uniprot protein.
    cmd.color('white', f'{gpcr}')
    resi_string =[str(x+1) for x in iface_in_human_seq if x >= 0]
    resi_string = '+'.join(resi_string)
    cmd.color('orange', f'{gpcr} and resi {resi_string}')

    outdir = f'generated_indices/{gpcr}'
    if not os.path.exists(outdir):
        os.makedirs(outdir)

    # Now go through each sequence in UNIREF50 for the GPCR and align it to the prealigned human model. 
    count_homo = 0.0
    for fn in os.listdir(curdir):
        if fn.endswith('seq.npy'):
            acc = fn.split('_')[0]
            seq = np.load(os.path.join(curdir, fn))
            # Ignore sequences if they are less than 70% of the human sequence
            if len(seq) > 0.70*len(human_seq):
                print ('######')
                print(f'Len human: {len(human_seq)} len homolog: {len(seq)} ')
                # For sanity, check that the sequence has X identity. Also store the identity. 
                # Perform the actual alignment to the template and check the identity of the sequence.
                iface_in_uniref_seq_ix, seq_identity, iface_seq = align_seqs(query_aln, seq, rinterface=iface_in_query_aln)

                # Open the data directory. 
                # Save the indices of the interface, the sequence identity of the entire sequence, and the sequence itself.
                np.save(os.path.join(outdir, acc+'_iface_indices.npy'), iface_in_uniref_seq_ix)
                np.save(os.path.join(outdir, acc+'_iface_seq.npy'), iface_seq)
                np.save(os.path.join(outdir, acc+'_seq_identity.npy'), seq_identity)



cmd.save('debug.pse')
