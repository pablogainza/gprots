import os 
import numpy as np
import sys
from IPython.core.debugger import set_trace
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score
results_dir = sys.argv[1]
preds= {}
gts = {}
for gprot in os.listdir(results_dir): 
    gprotdir = os.path.join(results_dir, gprot)
    preds[gprot] = {}
    gts[gprot] = {}
    for gpcr in os.listdir(gprotdir): 
        gpcrfn = os.path.join(gprotdir, gpcr)
        with open(gpcrfn) as f:
            result_line = f.readlines()[0]
            pred = float(result_line.split(',')[0].split(':')[1])
            gt = float(result_line.split(',')[1].split(':')[1])
            preds[gprot][gpcr] = pred
            gts[gprot][gpcr] = gt


overall_pred = []
overall_gt = []
for key in preds: 
    ypred = [preds[key][x] for x in preds[key]]
    ytrue = [gts[key][x] for x in gts[key]]
    overall_pred.append(ypred)
    overall_gt.append(ytrue)
    rocauc = roc_auc_score(ytrue, ypred)
    acc = accuracy_score(ytrue, np.round(ypred))
    bal_acc = balanced_accuracy_score(ytrue, np.round(ypred))
    print(f'G protein: {key} roc_auc: {rocauc:.3f}, accuracy: {acc:.3f}, balanced_accuracy: {bal_acc:.3f}')
ypred = np.concatenate(overall_pred)
ytrue = np.concatenate(overall_gt)

bal_acc = balanced_accuracy_score(ytrue, np.round(ypred))
rocauc = roc_auc_score(ytrue, ypred)
acc = accuracy_score(ytrue, np.round(ypred))
print(f'Overall performance: roc_auc: {rocauc:.3f}, accuracy: {acc:.3f}, balanced_accuracy: {bal_acc:.3f}')



        
