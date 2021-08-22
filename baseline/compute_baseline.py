# Compute a baseline on the ground truth using percent identity between proteins 
# We have ground truth for 48/49 proteins. 
# Split that into 8 validation and 40 test. 
# Do that multiple times and average the predictions. 
from IPython.core.debugger import set_trace
import numpy as np
from sklearn.metrics import roc_auc_score
import sys


# REad the ground truth. 
import pandas as pd
df = pd.read_csv('../ground_truth.csv')
index_start = sys.argv[1]

testset_list = []
with open(f'lists/testing{index_start}.txt') as f: 
    for line in f.readlines(): 
        testset_list.append(line.rstrip())

trainset_list = []
with open(f'lists/training{index_start}.txt') as f: 
    for line in f.readlines(): 
        trainset_list.append(line.rstrip())
 
trainset = df[df['HGNC'].isin(trainset_list)]
testset = df[df['HGNC'].isin(testset_list)]

trainset_uniprot_ids = trainset['UniprotAcc'].tolist()
print(f'Size of trainset: {len(trainset_uniprot_ids)}')
testset_uniprot_ids = testset['UniprotAcc'].tolist()
print(f'Size of testset: {len(trainset_uniprot_ids)}')

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


file_strong_act = open(f'output/baseline_strong_activation_{index_start}.csv','w+')
file_strong_act.write ('GPCR_NAME,gnaoAct,gnaqAct,gna15Act,gnas2Act,gnas13Act\n')
file_bin_amp= open(f'output/baseline_binary_amplitude_{index_start}.csv','w+')
file_bin_amp.write ('GPCR_NAME,gnaoAmp,gnaqAmp,gna15Amp,gnas2Amp,gnas13Amp\n')
file_3level_act= open(f'output/baseline_3_levels_activation.csv','w+')

strong_act_ypred = []
strong_act_ytrue = []
binary_amp_ypred = []
binary_amp_ytrue = []


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

    cur_trainset = trainset_uniprot_ids
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


    #pred = np.round(np.mean(strong_activation, axis=0))
    pred = np.mean(strong_activation, axis=0)
    pred= np.squeeze(pred)
    file_strong_act.write(f"{name},{pred[0]},{pred[1]},{pred[2]},{pred[3]},{pred[4]}\n")
    strong_act_ypred.append(pred)
    my_ytrue = np.array([row['gnaoAct'], row['gnaqAct'], row['gna15Act'], row['gnas2Act'], row['gnas13Act']]).astype(np.float)
    my_ytrue[my_ytrue < 30] = 0.0
    my_ytrue[my_ytrue >= 30] = 1.0
    strong_act_ytrue.append(my_ytrue)

    pred = np.round(np.mean(binary_amplitude, axis=0))
    pred= np.squeeze(pred)
    file_bin_amp.write(f"{name},{pred[0]},{pred[1]},{pred[2]},{pred[3]},{pred[4]}\n")
    binary_amp_ypred.append(pred)
    my_ytrue = np.array([row['gnaoAmp'], row['gnaqAmp'], row['gna15Amp'], row['gnas2Amp'], row['gnas13Amp']]).astype(np.float)
    my_ytrue[my_ytrue > 0] = 1.0
    binary_amp_ytrue.append(my_ytrue)

    pred = np.round(np.mean(three_level_activation, axis=0))
    pred= np.squeeze(pred)
    file_3level_act.write(f"{name},{pred[0]},{pred[1]},{pred[2]},{pred[3]},{pred[4]}\n")
    
strong_act_ytrue = np.array(strong_act_ytrue)
strong_act_ypred = np.array(strong_act_ypred)
binary_amp_ytrue = np.array(binary_amp_ytrue)
binary_amp_ypred = np.array(binary_amp_ypred)

# Print overall and per class ROC AUC. 
overall_roc_strong_act = roc_auc_score(np.concatenate(strong_act_ytrue), np.concatenate(strong_act_ypred))
overall_roc_binary_amp = roc_auc_score(np.concatenate(binary_amp_ytrue), np.concatenate(binary_amp_ypred))

with open(f'output/roc_auc_values_{index_start}.txt', 'w+') as f:
    f.write(f'Overall ROC AUC strong activation {overall_roc_strong_act:.3f}\n')
    for ix, gprot in enumerate(['gnaoAct', 'gnaqAct', 'gna15Act', 'gnas2Act', 'gnas13Act']):
        try:
            f.write(f'{gprot} {roc_auc_score(strong_act_ytrue[:,ix], strong_act_ypred[:,ix])}\n')    
        except: 
            f.write(f'{gprot} N/D\n')
    f.write(f'Overall ROC AUC binary amplitude {overall_roc_binary_amp:.3f}\n')
    for ix, gprot in enumerate(['gnaoAmp', 'gnaqAmp', 'gna15Amp', 'gnas2Amp', 'gnas13Amp']):
        try:
            f.write(f'{gprot} {roc_auc_score(binary_amp_ytrue[:,ix], binary_amp_ypred[:,ix])}\n')    
        except: 
            f.write(f'{gprot} N/D\n')


