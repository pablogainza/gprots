# Use matrix.txt to make clusters of strict splits, including testing, training and validation.
## Modified to make random splits. 
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import pandas as pd
from IPython.core.debugger import set_trace
import sys

output_index = sys.argv[1]

ground_truth = pd.read_csv('../ground_truth.csv')
classB_classC = ground_truth[(ground_truth['Class'] == 'C') | (ground_truth['Class'] == 'B')]
classB_classC = classB_classC['UniprotAcc'].tolist()

hgnc_A  = ground_truth[(ground_truth['Class'] == 'A')]['HGNC']
hgnc_A = hgnc_A.tolist()
np.random.shuffle(hgnc_A)
n = len(hgnc_A)

training = hgnc_A[:int(n*0.60)]
validation = hgnc_A[int(n*0.60):int(n*0.80)]
testing = hgnc_A[int(n*0.80):]


with open(f'lists/training{output_index}.txt', 'w+') as f:
    for name in training:
        f.write(name+'\n')
with open(f'lists/validation{output_index}.txt', 'w+') as f:
    for name in validation:
        f.write(name+'\n')
with open(f'lists/testing{output_index}.txt', 'w+') as f:
    for name in testing:
        f.write(name+'\n')

