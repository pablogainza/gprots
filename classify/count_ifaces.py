count_below4 = 0
with open('iface_distances.txt') as f: 
    for line in f.readlines(): 
        vals = line.rstrip().split(',')[1]
        if float(vals) <= 4.0:
            count_below4 += 1
            print(line.rstrip())
print(count_below4)
