import os
from IPython.core.debugger import set_trace
import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score
import pandas as pd
# Read the ground truth to get the same order. 
ground_truth = pd.read_csv('../ground_truth.csv')
test_gpcrs = ground_truth[(ground_truth['gnaoAct'] == '?') & (ground_truth['Class'] == 'A')]
test_gpcrs = test_gpcrs['HGNC'].tolist()
gprot_in_order = ['gnaoAct', 'gnaqAct', 'gna15Act', 'gnas2Act', 'gnas13Act']

# Read all validation results, collate, and get the ROC AUC. 
basedir = 'predictions_binary_act/'
gprots = os.listdir(basedir)
outdir = 'collated_results_binary_act/'
if not os.path.exists(outdir):
    os.makedirs(outdir)
with open(os.path.join(outdir, 'summary_statistics_validation_set.txt'), 'w+') as outf:
    for gprot in gprots: 
        ypred = []
        ytrue = []
        curdir = os.path.join(basedir,gprot)
        for fn in os.listdir(curdir):
            if fn.startswith('val'):
                with open(os.path.join(curdir,fn)) as f:
                    for line in f.readlines()[1:]:
                        fields = line.split(',')
                        ypred.append(float(fields[1]))
                        ytrue.append(float(fields[2]))
        print(f'ROC AUC {gprot} = {roc_auc_score(ytrue, ypred)}')
        outf.write(f'ROC AUC {gprot} = {roc_auc_score(ytrue, ypred):.3f}\n')
        print(f'Accuracy {gprot} = {accuracy_score(ytrue, np.round(ypred))}')
        outf.write(f'Accuracy {gprot} = {accuracy_score(ytrue, np.round(ypred)):.3f}\n')

# Read all test results, collate
basedir = 'predictions_binary_act/'
gprots = os.listdir(basedir)
outdir = 'collated_results_binary_act/'
outf = open(os.path.join(outdir, 'predictions_binary_act_regression.csv'), 'w+')
outf_r = open(os.path.join(outdir, 'predictions_binary_act_rounded.csv'), 'w+')

gprot_gpcr_pred = {}
for gprot in gprots: 
    curdir = os.path.join(basedir,gprot)
    gprot_gpcr_pred[gprot] = {}
    for fn in os.listdir(curdir):
        if fn.startswith('test'):
            with open(os.path.join(curdir,fn)) as f:
                for line in f.readlines()[1:]:
                    fields = line.split(',')
                    score = float(fields[1])
                    gpcr = fields[0]
                    if gpcr not in gprot_gpcr_pred[gprot]: 
                        gprot_gpcr_pred[gprot][gpcr]= []
                    gprot_gpcr_pred[gprot][gpcr].append(score)
outf.write(f'GPCR_NAME')
outf_r.write(f'GPCR_NAME')
for gprot in gprot_in_order: 
    outf.write(f',{gprot}')
    outf_r.write(f',{gprot}')
outf.write('\n')
outf_r.write('\n')
for gpcr in test_gpcrs:
    outf.write(f'{gpcr}')
    outf_r.write(f'{gpcr}')
    for gprot in gprot_in_order: 
        outf_r.write(f',{np.round(np.mean(gprot_gpcr_pred[gprot][gpcr])):.3f}')
        outf.write(f',{np.mean(gprot_gpcr_pred[gprot][gpcr]):.3f}')
    outf.write('\n')
    outf_r.write('\n')

outf.close()
outf_r.close()


# Finally parse the interpretation results for each gpcr
basedir = 'interpretation_binary_act/'
outdir = 'collated_results_binary_act/interpretations/'
if not os.path.exists(outdir):
    os.makedirs(outdir)

for gpcr in test_gpcrs:  
    iface_residues = {}
    for gprot in gprot_in_order:
        iface_residues[gprot] = {}
        for i in range(2,21):
            curfn = os.path.join(basedir,gprot,f'interpretation_{gpcr}_{i}')
            with open(curfn) as f: 
                for line in f.readlines():
                    resi = line.split(':')[0]
                    if resi not in iface_residues[gprot]:
                        iface_residues[gprot][resi] = []
                    iface_residues[gprot][resi].append(float(line.split(':')[1]))
    
    with open(os.path.join(outdir, f'interpretation_{gpcr}.csv'), 'w+') as outf:
        outf.write('gprot')
        for key in iface_residues[gprot_in_order[0]]:
            outf.write(f',{key}')
        outf.write('\n')
        for gprot in gprot_in_order:
            outf.write(f'{gprot}')
            for resi in iface_residues[gprot]:
                outf.write(f',{np.mean(iface_residues[gprot][resi]):.5f}')
            outf.write('\n')

