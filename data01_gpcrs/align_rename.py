#!/usr/local/Cellar/python@3.9/3.9.13_4/bin/python3.9
import sys
import os
sys.path.append('/usr/local/Cellar/pymol/2.4.0_3/libexec/lib/python3.9/site-packages')
from pymol import cmd
import os
import pandas as pd

df = pd.read_csv('../ground_truth.csv')
print(df['HGNC'].tolist())

for ix, row in df.iterrows():
    cmd.reinitialize()
    # load 7jvq
    cmd.load('7jvq.pdb', 'obj01')
    # Remove everything but chain R
    cmd.select('chainR', 'chain R and obj01')
    cmd.create('align_target', 'chainR')
    cmd.delete('obj01')
    print(row['HGNC'])

    if row['Prospective'] != '-': 
        pdbfn = f"input_pdbs/from_prospective/{row['Prospective']}"
        print(pdbfn)
        cmd.load(pdbfn, 'src')
        cmd.remove('src and not chain R')
    elif row['GPCRDB_refined'] != '-':
        pdbfn = f"input_pdbs/from_gpcrdb_refined/{row['GPCRDB_refined']}"
        print(pdbfn)
        cmd.load(pdbfn, 'src')
        cmd.remove('src and not chain R')
    elif row['PDB'] != '-':
        pdbfn = f"input_pdbs/from_pdb/{row['PDB']}"
        print(pdbfn)
        cmd.load(pdbfn, 'src')
    elif row['Baker'] != '-':
        pdbfn = f"input_pdbs/from_baker/{row['Baker']}"
        print(pdbfn)
        cmd.load(pdbfn, 'src')
    elif row['GPCRDB'] != '-':
        pdbfn = f"input_pdbs/from_gpcrdb/{row['GPCRDB']}"
        print(pdbfn)
        cmd.load(pdbfn, 'src')
    else:
        print(f"No pdb file entry for {row} ")
    # Remove not protein.
    cmd.remove('src and not polymer.protein')

    cmd.cealign('align_target', 'src', quiet=0)
#    print(f"RMSD {row['HGNC']} = {ret[0]}")
    cmd.alter('src', 'chain=\"R\"')
    cmd.save('aligned_renamed_pdbs/{}.pdb'.format(row['HGNC']), 'src')

