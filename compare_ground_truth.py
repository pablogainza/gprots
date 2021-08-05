ground_truth_file = 'ground_truth.csv'
our_results_file = 'classify/collated_results_binary_act/predictions_binary_act_regression.csv'
baseline_file = 'baseline/baseline_strong_activation.csv'

import pandas as pd 
from IPython.core.debugger import set_trace
from sklearn.metrics import roc_auc_score
gt = pd.read_csv(ground_truth_file)
ours = pd.read_csv(our_results_file)
baseline = pd.read_csv(baseline_file)

# Compute per-class ROC AUC
def compute_roc_auc_per_class(df, gt): 
    classnames = ['gnaoAct', 'gnaqAct', 'gna15Act', 'gnas2Act', 'gnas13Act']
    ytrue = []
    ypred = []
    for row in df.iterrows(): 
        for classname in classnames:  
            pred = row[1][classname]
            gene = row[1]['GPCR_NAME']
            gt_true = gt[gt['HGNC'] == gene][classname].values[0]
            if gt_true != '?':
                gt_true = float(gt_true)
                if gt_true < 30: 
                    gt_true = 0
                else: 
                    gt_true = 1
                   
                ytrue.append(float(gt_true))
                ypred.append(float(pred))
    print(roc_auc_score(ytrue, ypred))
    

compute_roc_auc_per_class(baseline, gt)
compute_roc_auc_per_class(ours, gt)
