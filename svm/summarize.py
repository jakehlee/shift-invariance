import sys
import numpy as np
import csv

if __name__ == "__main__":
    path = sys.argv[1]
    
    acc = []
    stddev = []
    with open(path, 'r') as f:
        r = csv.reader(f, delimiter=',')
        for line in r:
            acc.append(float(line[1]))
            stddev.append(float(line[2]))

    print(np.array(acc).mean())
    print(np.array(stddev).mean())
