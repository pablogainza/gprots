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
    def __init__(self, gpcrs, human_acc, ground_truth_gpcrs, batch_size=32, seqid_cutoff=0.5, human_only=False):
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
                if seq_id > seqid_cutoff:
                    # Load features only for the indices that are in the interface. 
                    feat = np.load(os.path.join(data_dir.format(gpcr), acc+'_feat.npy'))[iface_ix]
                    feat[zero_cols,:] = 0
                    self.X.append(feat)
                    ground_truth = ground_truth_gpcrs[gpcr]
                    ground_truth_array = np.zeros(3)
                    if ground_truth >30: 
                        ground_truth_array[0] = 1
                    elif ground_truth > 0 and ground_truth <30:
                        ground_truth_array[1] = 1
                    else:
                        ground_truth_array[2] = 1
                    self.Y.append(ground_truth_array)
                    self.accessions.append(ground_truth_gpcrs)
                    self.names.append(gpcr)
        
        self.batch_size = batch_size
        self.indexes = np.arange(len(self.Y))    
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
            
# Receive as a parameter the GPCR that will be the testing set. 
test_gpcr = [sys.argv[1]]
gprotein = sys.argv[2]
ignore_gpcr = 'PTGDR'

# Load the ground truth.
gtdf = pd.read_csv('../ground_truth.csv')

all_gpcrs = gtdf['GeneName'].tolist()
human_uniprotid = gtdf['UniprotAcc'].tolist()
amplitude = gtdf[gprotein].tolist()

groundtruth = {}
human_acc = {}
for ix, gpcr in enumerate(all_gpcrs):
    groundtruth[gpcr] = amplitude[ix]
    human_acc[gpcr] = human_uniprotid[ix]

all_ids = np.arange(len(all_gpcrs))
np.random.shuffle(all_ids)

n = len(all_ids)
training_ids = [x for x in all_ids[:38] if all_gpcrs[x] != test_gpcr[0] and all_gpcrs[x] != ignore_gpcr]
val_ids = [x for x in all_ids[38:] if all_gpcrs[x] != test_gpcr[0] and all_gpcrs[x] != ignore_gpcr]

training_gpcrs = [all_gpcrs[x] for x in training_ids]
val_gpcrs = [all_gpcrs[x] for x in val_ids]

# Define generator for validation and for training (possibly, mix ?) 
test_generator = DataGenerator(test_gpcr, human_acc, groundtruth, batch_size=8, seqid_cutoff=0.5, human_only=True)
training_generator = DataGenerator(training_gpcrs, human_acc, groundtruth, batch_size=8, seqid_cutoff=0.5)
val_generator = DataGenerator(val_gpcrs, human_acc, groundtruth, batch_size=8, seqid_cutoff=0.8)

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
model.add(Dense(16, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(3, activation = 'softmax'))

opt = optimizers.Adam(lr=LR)

model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_crossentropy', 'accuracy'])
#model.summary()

checkpoint = ModelCheckpoint(f'models/weights_learn_activation_3cat_{gprotein}_{test_gpcr[0]}.hdf5', monitor='val_loss', verbose=1, save_best_only=True, mode='min')
callbacks_list = [checkpoint]
history = model.fit_generator(generator=training_generator, epochs=100, validation_data=val_generator, callbacks=callbacks_list)

test_input, val = test_generator.__getitem__(0)
result = model.predict(test_input)
with open('predictions_activation_3cat/{}/{}.txt'.format(gprotein, test_gpcr[0]), 'w+') as f: 
    f.write(f'Pred: {result[0]}, gt: {val[0]}\n') 
    f.write(f"Training gpcrs: {','.join(training_gpcrs)}\n")
    f.write(f"Val gpcrs: {','.join(val_gpcrs)}\n")
        
#model.load_weights('models/weights_learn_activation_{}.hdf5'.format(cur_gpcr))

# Define the model (same one for neural network) 
    # Input: N, 1024
    # Fully connected: (N, 256)
    # Fully connected: (N, 128)
    # Fully connected: (N, 32)
    # Reshape N*32 or Max
    # Fully connected 1 with sigmoid

# Evaluate model on testing set.
