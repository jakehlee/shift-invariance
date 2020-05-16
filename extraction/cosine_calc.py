import sys, os
import numpy as np
import csv
import math

def usage():
    print("Usage: python cosine_calc.py dataset_dir")
    sys.exit(0)

def cosine_sim(A, B):
    return np.dot(A, B) / (np.linalg.norm(A) * np.linalg.norm(B))

def batch_cosine(A, B):
    # dot product
    top = np.multiply(A, B).sum(1)          # list of dot products
    A_norm = np.linalg.norm(A, axis=1)      # A norm
    B_norm = np.linalg.norm(B, axis=1)      # B norm
    bottom = A_norm * B_norm
    return top / bottom


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
    if len(sys.argv) != 2:
        usage()

    dataset_dir = sys.argv[1]
    listdir = os.listdir(dataset_dir)
    subdirs = []
    for d in listdir:
        if os.path.isdir(os.path.join(dataset_dir, d)):
            subdirs.append(d)

    for sd in subdirs:
        currdir = os.path.join(dataset_dir, sd)

        # find csvs in this directory
        filenames = os.listdir(currdir)
        csvs = [fn for fn in filenames if fn.endswith("csv")]

        for c in csvs:
            filepath = os.path.join(currdir, c)
            feats = import_features(filepath)
            
            side = int(math.sqrt(feats.shape[0]))
            tl = feats[0]
            tr = feats[side-1]
            cn = feats[int((side / 2 - 1) * side + (side / 2 - 1))]
            bl = feats[side * (side-1)]
            br = feats[-1]

            tl = np.tile(tl[2:], (feats.shape[0],1))
            tr = np.tile(tr[2:], (feats.shape[0],1))
            cn = np.tile(cn[2:], (feats.shape[0],1))
            bl = np.tile(bl[2:], (feats.shape[0],1))
            br = np.tile(br[2:], (feats.shape[0],1))

            cosines = np.zeros((feats.shape[0], 7))

            cosines[:,0] = feats[:,0]
            cosines[:,1] = feats[:,1]
            cosines[:,2] = batch_cosine(feats[:,2:], tl)
            cosines[:,3] = batch_cosine(feats[:,2:], tr)
            cosines[:,4] = batch_cosine(feats[:,2:], cn)
            cosines[:,5] = batch_cosine(feats[:,2:], bl)
            cosines[:,6] = batch_cosine(feats[:,2:], br)

            new_csv = os.path.join(currdir, c.split('.')[0]+'_cosines.csv')
            np.savetxt(new_csv, cosines, delimiter=",")
            print("Wrote to", new_csv)









