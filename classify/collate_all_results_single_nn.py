import os 
import sys
import pandas
import numpy as np
from sklearn.metrics import roc_auc_score, roc_curve
from IPython.core.debugger import set_trace
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

ix_range = np.arange(20,40)
basedir = 'predictions_binary_single_nn/'
act_or_amp = sys.argv[1]

ytrue = []
ypred = []
per_class_ytrue = {}
per_class_ypred = {}

experiments = {
                'DNNx1_PRETRAINx1xHUMANONLYx0' : '(Our method): DNN, pretraining',
                'DNNx1_PRETRAINx0xHUMANONLYx0' : 'DNN, no pretraining',
                'DNNx1_PRETRAINx1xHUMANONLYx1': 'DNN, pretrained, Human only',
}

for gprot in ['gnao', 'gnaq', 'gna15', 'gnas2', 'gnas13']: 
    legend = []


    for expkey in experiments.keys(): 
        print(experiments[expkey])
        per_class_ytrue[(gprot,expkey)] = []
        per_class_ypred[(gprot,expkey)] = []
        ytrue = []
        ypred = []
        for ix in ix_range:
            results = pandas.read_csv(os.path.join(basedir,act_or_amp,f'test_{ix}_{expkey}.csv'))
            p = results['ypred_'+gprot+act_or_amp].tolist()
            p = [float(x) for x in p]
            t = results['ytrue_'+gprot+act_or_amp].tolist()
            t = [float(x) for x in t]
            per_class_ytrue[(gprot, expkey)].extend(t)
            per_class_ypred[(gprot, expkey)].extend(p)

        ytrue = per_class_ytrue[(gprot, expkey)]
        ypred = per_class_ypred[(gprot, expkey)]
        fpr, tpr, _ = roc_curve(ytrue,ypred)
        roc_auc = roc_auc_score(ytrue, ypred)
        legend.append(f'{experiments[expkey]} ROC AUC = {roc_auc:.3f}')
        plt.plot(fpr,tpr)

    # Plot ROC curve for baseline
#    ytrue = np.load(f'../baseline/strong_act_{gprot}_baseline_ytrue.npy')
#    ypred = np.load(f'../baseline/strong_act_{gprot}_baseline_ypred.npy')
#    ytrue[ytrue < 30] = 0
#    ytrue[ytrue >=30] = 1
#    fpr, tpr, _ = roc_curve(ytrue, ypred)
#    roc_auc = roc_auc_score(ytrue, ypred)
#    legend.append(f'Baseline ROC AUC = {roc_auc:.3f}')
#    plt.plot(fpr,tpr)
        
    plt.legend(legend)
    plt.savefig(f'{gprot}_strong_{act_or_amp}.png')
    plt.close()

roc_auc = roc_auc_score(ytrue,ypred)
print(f'overall ROC AUC: {roc_auc}')

for gprot in per_class_ytrue:
    print(f'{gprot} roc auc: {roc_auc_score(per_class_ytrue[gprot], per_class_ypred[gprot])}')
