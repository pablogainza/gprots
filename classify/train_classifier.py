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
            if gpcr =='HTR2B':
                set_trace()
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
        print('Number of positives: {}; number of negatives: {} len(names): {}\n names: {}'.format(np.sum(self.Y ==1), np.sum(self.Y ==0), len(self.names), self.names))
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

all_generator= DataGenerator(all_train_gpcr, human_acc, groundtruth, batch_size=8, seqid_cutoff=0.8, human_only=True)
set_trace()

np.random.shuffle(all_train_gpcr)

training_gpcrs = all_train_gpcr[0:38]
val_gpcrs= all_train_gpcr[38:]
print(f'Total number of training gpcrs: {len(all_train_gpcr)}')



# Define generator for validation and for training (possibly, mix ?) 
training_generator = DataGenerator(training_gpcrs, human_acc, groundtruth, batch_size=8, seqid_cutoff=0.5)
val_generator = DataGenerator(val_gpcrs, human_acc, groundtruth, batch_size=8, seqid_cutoff=0.8)
print(f'validation: {len(val_gpcrs)} {val_gpcrs}')
val_human_generator = DataGenerator(val_gpcrs, human_acc, groundtruth, batch_size=128, seqid_cutoff=0.8, human_only=True)
print(f'training: {len(training_gpcrs)} {training_gpcrs}')
train_human_generator = DataGenerator(training_gpcrs, human_acc, groundtruth, batch_size=128, seqid_cutoff=0.8, human_only=True)
print(f'testing: {test_gpcr}')
test_generator = DataGenerator(test_gpcr, human_acc, groundtruth, batch_size=8, seqid_cutoff=0.5, human_only=True)

set_trace()
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

# Compute ROC AUC on best model's validation set. 
val_input, ytrue = val_human_generator.__getitem__(0)
result = model.predict(val_input)

set_trace()


test_input, val = test_generator.__getitem__(0)
result = model.predict(test_input)
with open('predictions/{}/{}.txt'.format(gprotein, test_gpcr[0]), 'w+') as f: 
    f.write(f'Pred: {result[0][0]}, gt: {val[0]}\n') 
    f.write(f"Training gpcrs: {','.join(training_gpcrs)}\n")
    f.write(f"Val gpcrs: {','.join(val_gpcrs)}\n")
        
