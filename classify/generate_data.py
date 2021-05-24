# Read the ground truth.

# Read the PDB of the template structure. 

# Get the sequence of the template structure.

# Compute the interface residues, size N, of the template structure and label them. 

# For each GPCR in our dataset 

    # Go through each sequence in UNIREF for the GPCR

        # Align to template sequence (use clustal?)  

        # for the N residues in the interface get the 'fingerprints'

        # Save a tensor of size  (N,1024) containing the input data for this training instance. 

        # If the UNIREF sequence is the corresponding one for human, save it as HUMAN.npy

        # Save the ground truth in numpy for easy access.
