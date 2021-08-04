from tensorflow.keras import optimizers
from IPython.core.debugger import set_trace
from sklearn.metrics import roc_auc_score
import pandas as pd

import tensorflow.keras as keras
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv1D
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import GlobalMaxPooling1D
from tensorflow.keras.layers import GlobalAveragePooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import LSTM

import numpy as np
import os
import sys

# Make a generator class. 
    # In each epoch we randomly select 50 sequences for training. 
    # Go through everyother data directory. 
    # Go through every training instance. 
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, gpcrs, human_acc, ground_truth_gpcrs, batch_size=32, seqid_cutoff=0.5, human_only=False, shuffle_indexes=True):
        '''
            Load all data into memory
        '''
        data_dir = '../uniref/{}/in_data/'
        indices_dir= 'generated_indices/{}/'
        self.X = [] 
        self.names = []
        self.accessions = []
        self.Y = []
        for gpcr in gpcrs:
            unique_ids = [x.split('_')[0] for x in os.listdir(indices_dir.format(gpcr)) if x.endswith('iface_indices.npy')] 
            if human_only: 
                selected_acc = [human_acc[gpcr]]
            else:
                np.random.shuffle(unique_ids)
                # pad with copies if size is less than 50
                if len(unique_ids) < 50:
                    unique_ids = np.concatenate([unique_ids, unique_ids, unique_ids], axis=0)
                selected_acc = [unique_ids[x] for x in range(50)]
                # always include the human one.
                selected_acc[0] = human_acc[gpcr]

            for acc in selected_acc:
                iface_ix = np.load(os.path.join(indices_dir.format(gpcr), acc+'_iface_indices.npy'))
                zero_cols = np.where(iface_ix < 0)[0]
                iface_ix[zero_cols] = 0
                seq_id = np.load(os.path.join(indices_dir.format(gpcr), acc+'_seq_identity.npy'))
                # Check that it passes the cutoff but always include the human accession.
                if seq_id > seqid_cutoff or acc == human_acc[gpcr]:
                    # Load features only for the indices that are in the interface. 
                    feat = np.load(os.path.join(data_dir.format(gpcr), acc+'_feat.npy'))[iface_ix]
                    feat[zero_cols,:] = 0
                    self.X.append(feat)
                    ground_truth = ground_truth_gpcrs[gpcr]
                    if ground_truth == '?':
                        ground_truth = 0
                    else:
                        ground_truth = float(ground_truth)
                    if ground_truth > 0: 
                        ground_truth = 1.0
                    self.Y.append(ground_truth)
                    self.accessions.append(ground_truth_gpcrs)
                    self.names.append(gpcr)
        self.Y = np.array(self.Y)
        print('Number of positives: {}; number of negatives: {} len(names): {}\n '.format(np.sum(self.Y ==1), np.sum(self.Y ==0), len(self.names)))
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.Y))    
        if shuffle_indexes: 
            np.random.shuffle(self.indexes)
                
    def __len__(self):
        'number of batches per epoch'
        return int(np.floor(len(self.indexes) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Data is augmented as it is sampled every time. 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        Xitem = np.array([self.X[ix] for ix in indexes])
        Yitem = np.array([self.Y[ix] for ix in indexes])

        return Xitem, Yitem
          
test_gpcr = []
with open('lists/testing01.txt') as f:
    for line in f.readlines():
        test_gpcr.append(line.rstrip())

all_train_gpcr = []
with open('lists/training01.txt') as f:
    for line in f.readlines():
        all_train_gpcr.append(line.rstrip())

gprotein = sys.argv[1]
run_id = sys.argv[2]

# Load the ground truth.
gtdf = pd.read_csv('../ground_truth.csv')

all_gpcrs = gtdf['HGNC'].tolist()
human_uniprotid = gtdf['UniprotAcc'].tolist()
amplitude = gtdf[gprotein].tolist()

groundtruth = {}
human_acc = {}
for ix, gpcr in enumerate(all_gpcrs):
    groundtruth[gpcr] = amplitude[ix]
    human_acc[gpcr] = human_uniprotid[ix]

np.random.shuffle(all_train_gpcr)

training_gpcrs = all_train_gpcr[0:38]
val_gpcrs= all_train_gpcr[38:]
print(f'Total number of training gpcrs: {len(all_train_gpcr)}')



# Define generator for validation and for training (possibly, mix ?) 
training_generator = DataGenerator(training_gpcrs, human_acc, groundtruth, batch_size=32, seqid_cutoff=0.5, shuffle_indexes=True,  human_only=False)
val_generator = DataGenerator(val_gpcrs, human_acc, groundtruth, batch_size=8, seqid_cutoff=0.8, shuffle_indexes=True, human_only=False)
print(f'validation: {len(val_gpcrs)}')
val_human_generator = DataGenerator(val_gpcrs, human_acc, groundtruth, batch_size=128, seqid_cutoff=0.8, human_only=True, shuffle_indexes=False)
print(f'training: {len(training_gpcrs)}')
train_human_generator = DataGenerator(training_gpcrs, human_acc, groundtruth, batch_size=128, seqid_cutoff=0.8, human_only=True, shuffle_indexes=False)
print(f'testing: {test_gpcr}')
test_generator = DataGenerator(test_gpcr, human_acc, groundtruth, batch_size=60, seqid_cutoff=0.5, human_only=True, shuffle_indexes=False)

LR = 0.0009 # maybe after some (10-15) epochs reduce it to 0.0008-0.0007
drop_out = 0.38
batch_dim = 64

model = Sequential()
model.add(Input(shape=(79,1024)))
model.add(Dropout(drop_out))
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(drop_out))
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(drop_out))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(drop_out))
# Reduce to one.
model.add(Dense(32, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(drop_out))
model.add(Dense(4, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(1, activation = 'sigmoid'))

opt = optimizers.Adam(lr=LR)

model.compile(optimizer=opt, loss='binary_crossentropy', metrics=['binary_crossentropy', 'accuracy'])
#model.summary()

checkpoint = ModelCheckpoint(f'models/weights_learn_amplitude_{gprotein}_{run_id}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history = model.fit_generator(generator=training_generator, epochs=32, validation_data=val_generator, callbacks=callbacks_list)

# Load best model
model.load_weights(f'models/weights_learn_amplitude_{gprotein}_{run_id}.hdf5')

# Compute human predictions on validation set.
val_input, ytrue = val_human_generator.__getitem__(0)
result = model.predict(val_input)

# Save validation results
if not os.path.exists(f'predictions_binary_amp/{gprotein}/'):
    os.makedirs(f'predictions_binary_amp/{gprotein}/')
with open('predictions_binary_amp/{}/val_{}.csv'.format(gprotein, run_id), 'w+') as f: 
    f.write('gene,ypred,ytrue\n')
    for ix in range(len(result)):
        f.write(f'{val_gpcrs[ix]},{result[ix][0]},{ytrue[ix]}\n') 

# Compute human predictions on test (hidden) set 
test_input, _ = test_generator.__getitem__(0)
result = model.predict(test_input)

with open('predictions_binary_amp/{}/test_{}.csv'.format(gprotein, run_id), 'w+') as f: 
    f.write('gene,ypred\n')
    for ix in range(len(result)):
        f.write(f'{test_gpcr[ix]},{result[ix][0]}\n') 

# Interpret the predictions.
if not os.path.exists(f'interpretation_binary_amp/{gprotein}/'):
    os.makedirs(f'interpretation_binary_amp/{gprotein}/')
training_human_embeddings,_ = train_human_generator.__getitem__(0)
# Now for the hidden set, go through every gpcr and replace each position by a randomly chosen embedding from a different protein
for gpcr_ix, gpcr in enumerate(test_gpcr):
    # Make a copy of the predictions for this gpcr and repeat it as many times as residues there are.
    interpret_copy = np.repeat(np.expand_dims(test_input[gpcr_ix],0), test_input.shape[1],axis=0)
    # replace each residue embedding input by a randomly chosen one from a member of the training set. 
    for residue_ix in range(interpret_copy.shape[1]): 
        random_training_gpcr_ix = np.random.randint(low=0,high=training_human_embeddings.shape[0])
        interpret_copy[residue_ix][residue_ix] = training_human_embeddings[random_training_gpcr_ix][residue_ix]

    # Evaluate in the NN 
    result_interpretation = model.predict(interpret_copy)

    # For every result, substract the prediction from the test_input. 
    delta_difference = result[gpcr_ix] - result_interpretation

    # Open the indices
    acc = human_acc[gpcr]
    indices_fn = f'generated_indices/{gpcr}/{acc}_iface_indices.npy'
    iface_indices = np.load(indices_fn)
    seq_fn = f'generated_indices/{gpcr}/{acc}_iface_seq.npy'
    iface_seq = np.load(seq_fn)

    # Save the results to a text file
    with open('interpretation_binary_amp/{}/interpretation_{}_{}'.format(gprotein, gpcr, run_id), 'w+') as f: 
        for ix in range(delta_difference.shape[0]):
            f.write(f'{iface_seq[ix]}{iface_indices[ix]}: {np.squeeze(delta_difference[ix])}\n')

        


