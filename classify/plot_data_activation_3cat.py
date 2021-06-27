import os 
import numpy as np
import sys
from IPython.core.debugger import set_trace
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, confusion_matrix
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
            pred = result_line.split(',')[0].split(':')[1]
            pred = pred.replace('[','')
            pred = pred.replace(']','')
            pred = [float(x) for x in pred.split()]
            
            gt = result_line.split(',')[1].split(':')[1]
            gt = gt.replace('[','')
            gt = gt.replace(']','')
            gt = [float(x) for x in gt.split()]

            preds[gprot][gpcr] = pred
            gts[gprot][gpcr] = gt

overall_pred = []
overall_gt = []
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt 
import pandas as pd
import seaborn as sns
mycmap = sns.color_palette("coolwarm")

for key in preds: 
    ypred = [np.argmax(preds[key][x]) for x in preds[key]]
    ytrue = [np.argmax(gts[key][x]) for x in gts[key]]
    conf_mat = confusion_matrix(ypred, ytrue)
    confdf = pd.DataFrame(conf_mat, columns=['Strong', 'Weak', 'No activation'], index=['Strong', 'Weak', 'No activation'])
    conf_map = sns.heatmap(confdf, cmap=mycmap, annot=True)
    print(f'Bincount {key} ytrue: {np.bincount(ytrue)}')

    plt.savefig(f'conf_map_{key}')
    plt.close()
    
# Read GPCRs in order.
gpcr_in_order = []
with open('lists/gpcr_list.txt') as f:
    for line in f.readlines():
        gpcr_in_order.append(line.split('.')[0].rstrip()+'.txt')

gt_data = []
pred_data = []
for gpcr in gpcr_in_order:
    cur_pred = [preds['gnaoAc'][gpcr], preds['gnaqAc'][gpcr], preds['gna15Ac'][gpcr], preds['gnas2Ac'][gpcr], preds['gnas13Ac'][gpcr]]
    cur_pred = [np.argmax(x) for x in cur_pred]
    cur_pred = [np.abs(x-2) for x in cur_pred]
    cur_gt = [gts['gnaoAc'][gpcr], gts['gnaqAc'][gpcr], gts['gna15Ac'][gpcr], gts['gnas2Ac'][gpcr], gts['gnas13Ac'][gpcr]]
    cur_gt = [np.argmax(x) for x in cur_gt]
    cur_gt = [np.abs(x-2) for x in cur_gt]
    gt_data.append(cur_gt)
    pred_data.append(cur_pred)

import pandas as pd
import seaborn as sns
plt.figure(figsize=(10,20))
gprots_order = ['Go', 'Gq', 'G15', 'G2', 'G13']
gpcr_for_print = [x.split('.')[0] for x in gpcr_in_order]
preddf = pd.DataFrame(pred_data, columns=gprots_order, index=gpcr_for_print)
mycmap = sns.color_palette("coolwarm")
sns.heatmap(preddf, vmin=0, vmax=2, cmap=mycmap, annot=True)
plt.title("Activation strength prediction")
plt.savefig('activation_3cat_pred.png')
plt.close()

import seaborn as sns
plt.figure(figsize=(10,20))
gprots_order = ['Go', 'Gq', 'G15', 'G2', 'G13']
gpcr_for_print = [x.split('.')[0] for x in gpcr_in_order]
gtdf = pd.DataFrame(np.round(gt_data), columns=gprots_order, index=gpcr_for_print)
mycmap = sns.color_palette("coolwarm")
sns.heatmap(gtdf, vmin=0, vmax=2, cmap=mycmap, annot=True)
plt.title("Activation strength ground truth")
plt.savefig('activation_3cat_gt.png')
