import os 
from IPython.core.debugger import set_trace
best_pred = {}
for gprot in os.listdir('predictions_binary_amp/'):
    best_pred[gprot] = {}
    for fn in os.listdir(f'predictions_binary_amp/{gprot}'):
        with open(f'predictions_binary_amp/{gprot}/{fn}') as f: 
            for line in f.readlines()[1:]: 
                fields = line.rstrip().split(',')
                gene = fields[0]
                mean = float(fields[1].replace('(',''))
                ytrue = float(fields[2].replace(')',''))
                stddev = float(fields[3])
                if gene not in best_pred[gprot]: 
                    best_pred[gprot][gene] = (mean, stddev, ytrue)
                else: 
                    if best_pred[gprot][gene][1] > stddev:
                        best_pred[gprot][gene] = (mean, stddev, ytrue)
                    
print('Protein,gnaoAmp,gnaqAmp,gna15Amp,gnas2Amp,gnas13Amp')
for prot in ['mAChRA', 'mAChRB', 'GHSR', 'DMDop1R1','DMDAMB','DMDop2R','DM5HT1A','DM5HT1B','DM5HT2A','DM5HT2B','DM5HT7','DMAdoR']:
    outline = f'{prot},'
    for key in ['gnaoAmp', 'gnaqAmp', 'gna15Amp', 'gnas2Amp', 'gnas13Amp']:
        outline += f'{best_pred[key][prot][0]:.3f},'
    print(outline)
            
