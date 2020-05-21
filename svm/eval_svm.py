import sys, os
import numpy as np
import csv
import math

from sklearn.model_selection import cross_val_score
from sklearn import svm

def usage():
    print("Usage: python cosine_calc.py dataset_dir layer logdir")
    sys.exit(0)

def import_features(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)

        raw = list(reader)

    # initialize array
    out = np.zeros((len(raw), len(raw[0]) + 1))

    for row in raw:
        label = row[0]
        y = int(label.split("_")[0])
        x = int(label.split("_")[1])
        feat_row = [y, x] + row[1:]
        index = int(y * (math.sqrt(len(raw))) + x)
        out[index] = feat_row

    return out


if __name__ == "__main__":
    if len(sys.argv) != 4:
        usage()

    dataset_dir = sys.argv[1]
    layer = sys.argv[2]
    logdir = sys.argv[3]

    if not os.path.isdir(logdir):
        os.mkdir(logdir)

    listdir = os.listdir(dataset_dir)
    subdirs = []
    for d in listdir:
        if os.path.isdir(os.path.join(dataset_dir, d)):
            subdirs.append(d)

    tb_stats = []
    lr_stats = []

    for sd in subdirs:

        currdir = os.path.join(dataset_dir, sd)

        # find csvs in this directory
        filenames = os.listdir(currdir)
        csvs = [fn for fn in filenames if fn.endswith(layer)]

        for c in csvs:
            filepath = os.path.join(currdir, c)
            feats = import_features(filepath)[:,2:]
            
            side = int(math.sqrt(feats.shape[0]))

            # top = 0, bot = 1
            topbot_labels = np.zeros(len(feats))
            topbot_labels[int(len(feats)/2):] = 1

            # left = 0, right = 1
            _label_temp = np.zeros(side)
            _label_temp[int(side/2):] = 1
            leftright_labels = np.tile(_label_temp, side)

            print("{} TB SVD...".format(sd))

            clf = svm.SVC(kernel='linear', C=1)
            tb_scores = cross_val_score(clf, feats, topbot_labels, cv=5)

            print("{} LR SVD...".format(sd))

            clf = svm.SVC(kernel='linear', C=1)
            lr_scores = cross_val_score(clf, feats, leftright_labels, cv=5)

            tb_stats.append([sd, tb_scores.mean(), tb_scores.std()])
            lr_stats.append([sd, lr_scores.mean(), lr_scores.std()])

    with open(os.path.join(logdir, 'tb_log.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerows(tb_stats)

    with open(os.path.join(logdir, 'lr_log.csv'), 'w') as f:
        wr = csv.writer(f)
        wr.writerows(lr_stats)









