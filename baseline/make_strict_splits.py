# Use matrix.txt to make clusters of strict splits, including testing, training and validation.
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from IPython.core.debugger import set_trace
import sys

output_index = sys.argv[1]

ground_truth = pd.read_csv('../ground_truth.csv')
classB_classC = ground_truth[(ground_truth['Class'] == 'C') | (ground_truth['Class'] == 'B')]
classB_classC = classB_classC['UniprotAcc'].tolist()


uid_list = []
protein_list = []
X = []
with open('matrix.txt') as f: 
    for line in f.readlines()[1:]:
        fields = line.split()
        uid = fields[0].split('|')[1]
        protein = fields[0].split('|')[2]
        uid_list.append(uid)
        
        protein_list.append(protein)
        dataline = [100.0/float(x) for x in fields[1:]]
        X.append(dataline)

classA_uid_list = []
classA_protein_list = []
classA_X = []

for row_ix, uid_row in enumerate(uid_list): 
    if uid_row not in classB_classC:
        classA_uid_list.append(uid_row)
        classA_protein_list.append(protein_list[row_ix])
        classA_X.append([])
        for col_ix, uid_col in enumerate(uid_list):  
            if uid_col not in classB_classC:
                classA_X[-1].append(X[row_ix][col_ix])
            
X = classA_X
protein_list = classA_protein_list
uid_list = classA_uid_list
               

clustering = AgglomerativeClustering(distance_threshold=10,linkage='single',n_clusters=None,compute_distances=True)
clustering.fit(X)
print(clustering.labels_)
for i in range(len(clustering.labels_)): 
    print(clustering.labels_[i], protein_list[i])
#print(f"Number of clusters: {len(clustering.labels_)}")
#print(f"Unique clusters: {len(np.unique(clustering.labels_))}")
#print(clustering.distances_)

# Split into training/testing
cluster_list = np.unique(clustering.labels_)
np.random.shuffle(cluster_list)

# Training will be clusters adding to < 60% of the set, validation < 20% and testing the remaining ones.
# Validation: 
n = len(clustering.labels_)
validation_labels = []
for clust in cluster_list: 
    len_set = len([x for x in clustering.labels_ if x in validation_labels or x == clust])
    if len_set < n*0.20:
        validation_labels.append(clust)
    elif len_set < n*0.30:
        validation_labels.append(clust)
        break
    else:
        break

# Training will be clusters adding to < 60% of the set, validation < 20% and testing the remaining ones.
training_labels = []
for clust in cluster_list: 
    if clust not in validation_labels:
        len_trainset = len([x for x in clustering.labels_ if x in training_labels or x == clust])
        if len_trainset < n*0.60:
            training_labels.append(clust)
        elif len_trainset < n*0.70:
            training_labels.append(clust)
            break
        else:
            break

# Testing: 
testing_labels = []
for clust in cluster_list: 
    if clust not in training_labels and clust not in validation_labels: 
        testing_labels.append(clust)

# Find the maximum identity between training and testing. 
for ix_test, name_test in enumerate(protein_list): 
    if clustering.labels_[ix_test] in testing_labels: 
        # Max id was inverted before, so here we search in the opposite direction.
        max_id = float('inf')
        max_id_name = ''
        for ix_train, name_train in enumerate(protein_list):  
            if clustering.labels_[ix_train] in training_labels: 
                # Max id was inverted before, so here we search in the opposite direction.
                if X[ix_test][ix_train] < max_id: 
                    max_id = X[ix_test][ix_train]
                    max_id_name = name_train
        print(f'Most similar to {name_test} is {max_id_name} with {1.0/max_id:.2f}')

count_test = 0 
count_train = 0
count_val = 0

with open(f'lists/testing{output_index}.txt', 'w+') as f:
    for ix_test, name_test in enumerate(protein_list): 
        if clustering.labels_[ix_test] in testing_labels: 
            # convert name_test to HGNC
            hgnc = ground_truth[ground_truth['UniprotName'] == name_test] 
            hgnc = hgnc['HGNC']
            f.write(hgnc.values[0]+'\n')
            count_test +=1

with open(f'lists/training{output_index}.txt', 'w+') as f:
    for ix_train, name_train in enumerate(protein_list): 
        if clustering.labels_[ix_train] in training_labels: 
            hgnc = ground_truth[ground_truth['UniprotName'] == name_train] 
            hgnc = hgnc['HGNC']
            f.write(hgnc.values[0]+'\n')
            count_train +=1

with open(f'lists/validation{output_index}.txt', 'w+') as f:
    for ix_val, name_val in enumerate(protein_list): 
        if clustering.labels_[ix_val] in validation_labels: 
            hgnc = ground_truth[ground_truth['UniprotName'] == name_val] 
            hgnc = hgnc['HGNC']
            f.write(hgnc.values[0]+'\n')
            count_val +=1

print(f'Count test: {count_test}')
print(f'Count train: {count_train}')
print(f'Count val: {count_val}')

