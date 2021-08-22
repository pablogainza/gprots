import pandas as pd 
import numpy as np 
from sklearn.metrics import roc_auc_score
from IPython.core.debugger import set_trace

myrange = np.arange(20,39)
gt = pd.read_csv('../ground_truth.csv')
class_A = gt[gt['Class'] == 'A']

gprots = ['gnaoAct','gnaqAct','gna15Act','gnas2Act','gnas13Act']
ypred = []
ytrue = []
for ix in myrange: 
    with open(f'output/baseline_strong_activation_{ix:02}.csv') as f:
        for line in f.readlines()[1:]:
            fields = line.split(',')
            gene = fields[0]
            p = [float(x) for x in fields[1:]]
            ypred.append(p)
            row = class_A[class_A['HGNC'] == gene]
            t = [float(x) for x in row[gprots].values[0]]
            ytrue.append(t)

ytrue = np.array(ytrue)
ypred = np.array(ypred)

for ix, gprot in enumerate(gprots): 
    myytrue = ytrue[:,ix].copy()
    myypred = ypred[:,ix].copy()
    myytrue[myytrue <30] = 0
    myytrue[myytrue >=30] = 1
    roc_auc = roc_auc_score(myytrue, myypred)
    print(f'{gprot}: {roc_auc}')

ypred = np.concatenate(ypred)
ytrue = np.concatenate(ytrue)
ytrue[ytrue < 30] = 0
ytrue[ytrue >= 30] = 1

print(roc_auc_score(ytrue, ypred))

