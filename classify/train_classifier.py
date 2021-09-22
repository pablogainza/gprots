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
from sklearn.preprocessing import OneHotEncoder

import numpy as np
import os
import sys

aa_to_int = {
    'A': 0,
    'C': 1,
    'D': 2,
    'E': 3,
    'F': 4,
    'G': 5,
    'H': 6,
    'I': 7,
    'K': 8,
    'L': 9,
    'M': 10,
    'N': 11,
    'P': 12,
    'Q': 13,
    'R': 14,
    'S': 15,
    'T': 16,
    'V': 17,
    'W': 18,
    'Y': 19,
    'X': 20
}


# Make a generator class. 
    # In each epoch we randomly select 50 sequences for training. 
    # Go through everyother data directory. 
    # Go through every training instance. 
class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, gpcrs, human_acc, ground_truth_gpcrs, use_pretrained_embeddings=False, iface_cutoff=4, batch_size=32, seqid_cutoff=0.5, human_only=False, shuffle_indexes=True):
        '''
            Load all data into memory
        '''
        # Load the distances of each residue to the target. 
        iface_dists = []
        with open('iface_distances.txt') as f: 
            for line in f.readlines(): 
                iface_dists.append(float(line.rstrip().split(',')[1]))
        iface_dists = np.array(iface_dists)

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
                ## Remove indices those that are farther than iface_cutoff from the Gprotein
                iface_ix = iface_ix[np.where(iface_dists < iface_cutoff)[0]]
                zero_cols = np.where(iface_ix < 0)[0]
                iface_ix[zero_cols] = 0
                seq_id = np.load(os.path.join(indices_dir.format(gpcr), acc+'_seq_identity.npy'))
                # Check that it passes the cutoff but always include the human accession.
                if seq_id > seqid_cutoff or acc == human_acc[gpcr]:
                    # Load features only for the indices that are in the interface. 
                    if use_pretrained_embeddings:
                        feat = np.load(os.path.join(data_dir.format(gpcr), acc+'_feat.npy'))[iface_ix]
                        feat[zero_cols,:] = 0
                    else:
                        seq = np.load(os.path.join(data_dir.format(gpcr), acc+'_seq.npy'))[iface_ix]
                        seq = [aa_to_int[x] for x in seq]
                        feat = np.zeros((len(seq), len(aa_to_int.keys())))
                        # one hot encoding.
                        for i in range(len(seq)): 
                            feat[i,seq[i]] = 1.0
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

    def __getInputShape__ (self):
        return self.X[0].shape

    def __getitem__(self, index):
        'Generate one batch of data'
        # Data is augmented as it is sampled every time. 
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Generate data
        Xitem = np.array([self.X[ix] for ix in indexes])
        Yitem = np.array([self.Y[ix] for ix in indexes])

        return Xitem, Yitem

gprotein = sys.argv[1]
run_id = sys.argv[2]
use_DNN = int(sys.argv[3])
use_pretrained_embeddings = int(sys.argv[4]) 
use_human_only = int(sys.argv[5])
out_fn = f'predictions_binary_amp/{gprotein}/test_{run_id}_DNNx{use_DNN}_PRETRAINx{use_pretrained_embeddings}xHUMANONLYx{use_human_only}.csv'
if not os.path.exists(f'predictions_binary_amp/{gprotein}'):
    os.makedirs(f'predictions_binary_amp/{gprotein}')
          
test_gpcr = []
with open(f'lists/testing{int(run_id):02}.txt') as f:
    for line in f.readlines():
        test_gpcr.append(line.rstrip())

all_train_gpcr = []
with open(f'lists/training{int(run_id):02}.txt') as f:
    for line in f.readlines():
        all_train_gpcr.append(line.rstrip())

all_val_gpcr = []
with open(f'lists/validation{int(run_id):02}.txt') as f:
    for line in f.readlines():
        all_val_gpcr.append(line.rstrip())


# Load the ground truth.
gtdf = pd.read_csv('../ground_truth.csv')

all_gpcrs = gtdf['HGNC'].tolist()
human_uniprotid = gtdf['UniprotAcc'].tolist()
activation = gtdf[gprotein].tolist()

groundtruth = {}
human_acc = {}
for ix, gpcr in enumerate(all_gpcrs):
    groundtruth[gpcr] = activation[ix]
    human_acc[gpcr] = human_uniprotid[ix]

np.random.shuffle(all_train_gpcr)

training_gpcrs = all_train_gpcr
val_gpcrs= all_val_gpcr
print(f'Total number of training gpcrs: {len(all_train_gpcr)}')

# Define generator for validation and for training (possibly, mix ?) 
training_generator = DataGenerator(training_gpcrs, human_acc, groundtruth, use_pretrained_embeddings=use_pretrained_embeddings, batch_size=32, seqid_cutoff=0.6, shuffle_indexes=True,  human_only=use_human_only)
val_generator = DataGenerator(val_gpcrs, human_acc, groundtruth, use_pretrained_embeddings=use_pretrained_embeddings, batch_size=8, seqid_cutoff=0.8, shuffle_indexes=True, human_only=use_human_only)
print(f'validation: {len(val_gpcrs)}')
test_generator = DataGenerator(test_gpcr, human_acc, groundtruth, use_pretrained_embeddings=use_pretrained_embeddings, batch_size=128, seqid_cutoff=0.5, human_only=True, shuffle_indexes=False)

if use_DNN:
    LR = 0.0009 
    drop_out = 0.38
    input_shape = training_generator.__getInputShape__()

    model = Sequential()
    model.add(Input(shape=(input_shape)))
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

    checkpoint = ModelCheckpoint(f'models/weights_learn_amplitude_{gprotein}_{run_id}_{use_DNN}_{use_pretrained_embeddings}_{use_human_only}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
    callbacks_list = [checkpoint]
    history = model.fit_generator(generator=training_generator, epochs=100, validation_data=val_generator, callbacks=callbacks_list)

    # Load best model
    model.load_weights(f'models/weights_learn_amplitude_{gprotein}_{run_id}_{use_DNN}_{use_pretrained_embeddings}_{use_human_only}.hdf5')

    # Compute human predictions on test (hidden) set 
    test_input, ytrue = test_generator.__getitem__(0)
    result = model.predict(test_input)

    with open(out_fn, 'w+') as f: 
        f.write('gene,ypred\n')
        for ix in range(len(result)):
            f.write(f'{test_gpcr[ix]},{result[ix][0],ytrue[ix]}\n') 


        
# Logistic regression, for comparison purposes.
else:
    from sklearn.linear_model import LogisticRegression
    X = []
    Y = []
    for k in range(training_generator.__len__()):
        train_input, ytrue = training_generator.__getitem__(k)
        X.append(train_input)
        Y.append(ytrue)
    X = np.concatenate(X, axis=0)
    Y = np.concatenate(Y, axis=0)
    X = np.reshape(X, (X.shape[0], X.shape[1]*X.shape[2]))
    clf = LogisticRegression().fit(X,Y)

    # Compute human predictions on test (hidden) set 
    test_input, ytrue = test_generator.__getitem__(0)
    test_input = np.reshape(test_input, (test_input.shape[0], test_input.shape[1]*test_input.shape[2]))
    result = clf.predict(test_input)

    with open(out_fn, 'w+') as f: 
        f.write('gene,ypred\n')
        for ix in range(len(result)):
            f.write(f'{test_gpcr[ix]},{result[ix],ytrue[ix]}\n') 

