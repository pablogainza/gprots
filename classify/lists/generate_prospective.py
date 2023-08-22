import numpy as np 
all_training = []
with open ('full_list_training_validation.txt') as f: 
    for line in f.readlines(): 
        all_training.append(line.rstrip())


for i in range(40,50):
    np.random.shuffle(all_training)
    with open(f'training{i}.txt', 'w+') as f:
        for gpcr in all_training[0:84]:
            f.write(gpcr+'\n')
    with open(f'validation{i}.txt', 'w+') as f:
        for gpcr in all_training[84:len(all_training)]:
            f.write(gpcr+'\n')
    
    with open(f'testing{i}.txt', 'w+') as f:
        f.write('mAChRA\n')
        f.write('mAChRB\n')
        f.write('GHSR\n')
        f.write('DMDop1R1\n')
        f.write('DMDAMB\n')
        f.write('DMDop2R\n')
        f.write('DM5HT1A\n')
        f.write('DM5HT1B\n')
        f.write('DM5HT2A\n')
        f.write('DM5HT2B\n')
        f.write('DM5HT7\n')
        f.write('DMAdoR\n')
        for gpcr in all_training:
            f.write(gpcr+'\n')
