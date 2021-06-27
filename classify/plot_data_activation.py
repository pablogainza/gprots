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

import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
# Read GPCRs in order. 
gpcr_in_order = []
with open('lists/gpcr_list.txt') as f: 
    for line in f.readlines(): 
        gpcr_in_order.append(line.split('.')[0].rstrip()+'.txt')

gt_data = []
pred_data = []
for gpcr in gpcr_in_order:
    pred_data.append([preds['gnaoAc'][gpcr], preds['gnaqAc'][gpcr], preds['gna15Ac'][gpcr], preds['gnas2Ac'][gpcr], preds['gnas13Ac'][gpcr]])
    gt_data.append([gts['gnaoAc'][gpcr], gts['gnaqAc'][gpcr], gts['gna15Ac'][gpcr], gts['gnas2Ac'][gpcr], gts['gnas13Ac'][gpcr]])


import pandas as pd
import seaborn as sns
plt.figure(figsize=(10,20))
gprots_order = ['Go', 'Gq', 'G15', 'G2', 'G13']
gpcr_for_print = [x.split('.')[0] for x in gpcr_in_order]
preddf = pd.DataFrame(pred_data, columns=gprots_order, index=gpcr_for_print)
mycmap = sns.color_palette("coolwarm")
sns.heatmap(preddf, vmin=0, vmax=1, cmap=mycmap, annot=True)
plt.title("Binary activation prediction")
plt.savefig('activation_pred.png')
plt.close()

import seaborn as sns
plt.figure(figsize=(10,20))
gprots_order = ['Go', 'Gq', 'G15', 'G2', 'G13']
gpcr_for_print = [x.split('.')[0] for x in gpcr_in_order]
gtdf = pd.DataFrame(np.round(gt_data), columns=gprots_order, index=gpcr_for_print)
mycmap = sns.color_palette("coolwarm")
sns.heatmap(gtdf, vmin=0, vmax=1, cmap=mycmap, annot=True)
plt.title("Binary activation ground truth")
plt.savefig('activation_gt.png')
