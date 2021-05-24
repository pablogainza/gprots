# Receive as a parameter the GPCR that will be the testing set. 

# Make a generator class. 
    # In each epoch we randomly select 50 sequences for training. 
    # Go through everyother data directory. 
    # Go through every training instance. 

# Define generator for validation and for training (possibly, mix ?) 

# Define the model (same one for neural network) 
    # Input: N, 1024
    # Fully connected: (N, 256)
    # Fully connected: (N, 128)
    # Fully connected: (N, 32)
    # Reshape N*32 or Max
    # Fully connected 1 with sigmoid

# Evaluate model on testing set.
