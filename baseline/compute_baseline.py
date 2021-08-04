# Compute a baseline on the ground truth using percent identity between proteins 
# We have ground truth for 48/49 proteins. 
# Split that into 8 validation and 40 test. 
# Do that multiple times and average the predictions. 
from IPython.core.debugger import set_trace
import numpy as np


# REad the ground truth. 
import pandas as pd
df = pd.read_csv('../ground_truth.csv')

testset = df[df['gnaoAmp'] == '?']
trainset = df[df['gnaoAmp'] != '?']

trainset_uniprot_ids = trainset['UniprotAcc'].tolist()
print(f'Size of trainset: {len(trainset_uniprot_ids)}')
testset_uniprot_ids = testset['UniprotAcc'].tolist()

# Read identity matrix between train and test set. 
uid_test_to_train = {}
with open('matrix.txt') as f: 
    all_uid = []
    data = []
    for line in f.readlines()[1:]:
        fields = line.split()
        uid = fields[0].split('|')[1]
        all_uid.append(uid)
        dataline = [float(x) for x in fields[1:]]
        data.append(dataline)

    for ix, uid in enumerate(all_uid):
        if uid in testset_uniprot_ids:
            uid_test_to_train[uid] = {}
            for ix2, uid2 in enumerate(all_uid):
                if uid2 in trainset_uniprot_ids: 
                    uid_test_to_train[uid][uid2] = data[ix][ix2]


file_strong_act = open('baseline_strong_activation.csv','w+')
file_strong_act.write ('GPCR_NAME,gnaoAct,gnaqAct,gna15Act,gnas2Act,gnas13Act\n')
file_bin_amp= open('baseline_binary_amplitude.csv','w+')
file_bin_amp.write ('GPCR_NAME,gnaoAmp,gnaqAmp,gna15Amp,gnas2Amp,gnas13Amp\n')
file_3level_act= open('baseline_3_levels_activation.csv','w+')

for iiii, row in testset.iterrows():
    uid = row['UniprotAcc']
    name = row['HGNC']
    classname = row['Class']
    if classname != 'A':
        continue
    # prediction task 1: predict amplitude greater than zero. 
    binary_amplitude = []
    # prediction task 2: predict whether this row has strong activation or not. 
    strong_activation = []
    # prediction task 3: predict three levels of activation
    three_level_activation = []

    # To simulate our NN, we do 10 runs with different trainset uniprot ids, removing 8 each time, and average the predictions.
    for run in range(10):
        np.random.shuffle(trainset_uniprot_ids)
        cur_trainset = trainset_uniprot_ids[:41]
        # Find the most similar sequence in the trainset. 
        max_identity = 0.0
        most_similar = ''
        for uidtrain in cur_trainset:
            if uid_test_to_train[uid][uidtrain] > max_identity:
                most_similar = uidtrain
                max_identity = uid_test_to_train[uid][uidtrain] 
        trainrow = trainset[trainset['UniprotAcc'] == most_similar]

        amp_bin = np.array([trainrow['gnaoAmp'], trainrow['gnaqAmp'], trainrow['gna15Amp'], trainrow['gnas2Amp'], trainrow['gnas13Amp']]).astype(np.float)
        act_strong = np.array([trainrow['gnaoAct'], trainrow['gnaqAct'], trainrow['gna15Act'], trainrow['gnas2Act'], trainrow['gnas13Act']]).astype(np.float)
        act_3level= np.array([trainrow['gnaoAct'], trainrow['gnaqAct'], trainrow['gna15Act'], trainrow['gnas2Act'], trainrow['gnas13Act']]).astype(np.float)

        amp_bin[amp_bin > 0]  = 1.0

        act_strong[act_strong <30] = 0 
        act_strong[act_strong >= 30] = 1.0

        act_3level[(act_3level < 30) & (act_3level > 0)] = 1.0        
        act_3level[act_3level >= 30] = 2.0        

        binary_amplitude.append(amp_bin)
        strong_activation.append(act_strong)
        three_level_activation.append(act_3level)

        print(f"Most similar to {row['HGNC']} is {trainrow['HGNC'].tolist()[0]} with {max_identity}")


    pred = np.round(np.mean(strong_activation, axis=0))
    pred= np.squeeze(pred)
    file_strong_act.write(f"{name},{pred[0]},{pred[1]},{pred[2]},{pred[3]},{pred[4]}\n")

    pred = np.round(np.mean(binary_amplitude, axis=0))
    pred= np.squeeze(pred)
    file_bin_amp.write(f"{name},{pred[0]},{pred[1]},{pred[2]},{pred[3]},{pred[4]}\n")

    pred = np.round(np.mean(three_level_activation, axis=0))
    pred= np.squeeze(pred)
    file_3level_act.write(f"{name},{pred[0]},{pred[1]},{pred[2]},{pred[3]},{pred[4]}\n")
    
