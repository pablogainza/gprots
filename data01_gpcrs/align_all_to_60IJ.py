#!/usr/local/Cellar/python@3.9/3.9.2_2/bin/python3
import sys
import os
sys.path.append('/usr/local/Cellar/pymol/2.4.0_3/libexec/lib/python3.9/site-packages')
from pymol import cmd
import os

pdbfn = os.path.basename(sys.argv[1])

# load 6OIJ
cmd.load('../gproteins/6OIJ_G13/6OIJ.pdb', 'obj01')
# Remove everything but chain R
cmd.select('chainR', 'chain R and obj01')
cmd.create('align_target', 'chainR')
cmd.delete('obj01')
cmd.load(pdbfn, 'src')
cmd.align('src', 'align_target')
cmd.alter('src', 'chain=\"R\"')
cmd.save('aligned_pdbs/{}'.format(pdbfn), 'src')

