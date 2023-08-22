import pandas as pd 
import numpy as np 
from sklearn.metrics import roc_auc_score
from IPython.core.debugger import set_trace

myrange = np.arange(20,40)
gt = pd.read_csv('../ground_truth.csv')
class_A = gt[gt['Class'] == 'A']

gprots = ['gnaoAmp','gnaqAmp','gna15Amp','gnas2Amp','gnas13Amp']
ypred = []
ytrue = []
for ix in myrange: 
    with open(f'output/baseline_binary_amplitude_{ix:02}.csv') as f:
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
    np.save(f'binary_amp_{gprot}_baseline_ytrue', myytrue)
    np.save(f'binary_amp_{gprot}_baseline_ypred', myypred)

ypred = np.concatenate(ypred)
ytrue = np.concatenate(ytrue)
ytrue[ytrue < 30] = 0
ytrue[ytrue >= 30] = 1

print(roc_auc_score(ytrue, ypred))

