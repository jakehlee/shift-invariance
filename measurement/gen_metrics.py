import sys, os
import numpy as np
import csv
import math
import time
import matplotlib.pyplot as plt

MODEL = "MobileNet V2 AA"
LAYER = "fc"
BG = "white"

def usage():
    print("Usage: python gen_metrics.py dataset_dir expname")
    sys.exit(0)

def import_cosines(filepath):
    with open(filepath, 'r') as f:
        reader = csv.reader(f)

        raw = np.array(list(reader)).astype(np.float)
        print(raw.shape)

    return raw[:,:2], raw[:,2], raw[:,3], raw[:,4], raw[:,5], raw[:,6]

if __name__ == "__main__":
    if len(sys.argv) != 3:
        usage()

    dataset_dir = sys.argv[1]
    expname = sys.argv[2]

    if not os.path.isdir(expname):
        os.mkdir(expname)

    listdir = os.listdir(dataset_dir)
    subdirs = []
    for d in listdir:
        if os.path.isdir(os.path.join(dataset_dir, d)):
            subdirs.append(d)

    tl_avg = 0
    tr_avg = 0
    c_avg = 0
    bl_avg = 0
    br_avg = 0
    count = 0

    tl_sums = []
    tr_sums = []
    c_sums = []
    bl_sums = []
    br_sums = []
    
    for sd in subdirs:
        currdir = os.path.join(dataset_dir, sd)

        # find cosine csv in this directory
        filenames = os.listdir(currdir)
        csvs = [fn for fn in filenames if fn.endswith("cosines.csv")]

        for c in csvs:
            filepath = os.path.join(currdir, c)
            print(filepath)
            coord, tl, tr, c, bl, br = import_cosines(filepath)
            
            print(count)
            tl_avg = (tl_avg * count + tl) / (count + 1)
            tr_avg = (tr_avg * count + tr) / (count + 1)
            c_avg = (c_avg * count + c) / (count + 1)
            bl_avg = (bl_avg * count + bl) / (count + 1)
            br_avg = (br_avg * count + br) / (count + 1)
            count += 1

            tl_sums.append(np.sum(tl))
            tr_sums.append(np.sum(tr))
            c_sums.append(np.sum(c))
            bl_sums.append(np.sum(bl))
            br_sums.append(np.sum(br))
    
    print(tl_avg.shape)
    side = int(np.sqrt(tl_avg.shape[0]))
    tl_avg = tl_avg.reshape((side,side))
    tr_avg = tr_avg.reshape((side,side))
    c_avg = c_avg.reshape((side, side))
    bl_avg = bl_avg.reshape((side, side))
    br_avg = br_avg.reshape((side, side))

    with open(os.path.join(expname, "log.txt"), 'w') as f:
        f.write("TL\n")
        f.write("mean: ")
        f.write(str(np.mean(tl_sums)))
        f.write("\n")
        f.write("stddev: ")
        f.write(str(np.std(tl_sums)))
        f.write("\n")
        
        f.write("TR\n")
        f.write("mean: ")
        f.write(str(np.mean(tr_sums)))
        f.write("\n")
        f.write("stddev: ")
        f.write(str(np.std(tr_sums)))
        f.write("\n")

        f.write("C\n")
        f.write("mean: ")
        f.write(str(np.mean(c_sums)))
        f.write("\n")
        f.write("stddev: ")
        f.write(str(np.std(c_sums)))
        f.write("\n")

        f.write("BL\n")
        f.write("mean: ")
        f.write(str(np.mean(bl_sums)))
        f.write("\n")
        f.write("stddev: ")
        f.write(str(np.std(bl_sums)))
        f.write("\n")

        f.write("BR\n")
        f.write("mean: ")
        f.write(str(np.mean(br_sums)))
        f.write("\n")
        f.write("stddev: ")
        f.write(str(np.std(br_sums)))
        f.write("\n")
    
    LLIM = 0.88

    fig, ax = plt.subplots()
    pos = ax.imshow(tl_avg, cmap='hot')
    ax.set_title('Average Cosine Similarity of Shifted Object Features\n {} {} on 200 patches on {}, TOP LEFT'.format(MODEL, LAYER, BG))
    pos.set_clim(LLIM, 1)
    fig.colorbar(pos)
    plt.savefig(os.path.join(expname,"tl.png"))

    fig, ax = plt.subplots()
    pos = ax.imshow(tr_avg, cmap='hot')
    ax.set_title('Average Cosine Similarity of Shifted Object Features\n {} {} on 200 patches on {}, TOP RIGHT'.format(MODEL, LAYER, BG))
    fig.colorbar(pos)
    plt.savefig(os.path.join(expname,"tr.png"))

    fig, ax = plt.subplots()
    pos = ax.imshow(c_avg, cmap='hot')
    ax.set_title('Average Cosine Similarity of Shifted Object Features\n {} {} on 200 patches on {}, CENTER'.format(MODEL, LAYER, BG))
    pos.set_clim(LLIM, 1)
    fig.colorbar(pos)
    plt.savefig(os.path.join(expname,"c.png"))

    fig, ax = plt.subplots()
    pos = ax.imshow(bl_avg, cmap='hot')
    ax.set_title('Average Cosine Similarity of Shifted Object Features\n {} {} on 200 patches on {}, BOTTOM LEFT'.format(MODEL, LAYER, BG))
    pos.set_clim(LLIM, 1)
    fig.colorbar(pos)
    plt.savefig(os.path.join(expname,"bl.png"))

    fig, ax = plt.subplots()
    pos = ax.imshow(br_avg, cmap='hot')
    ax.set_title('Average Cosine Similarity of Shifted Object Features\n {} {} on 200 patches on {}, BOTTOM RIGHT'.format(MODEL, LAYER, BG))
    pos.set_clim(LLIM, 1)
    fig.colorbar(pos)
    plt.savefig(os.path.join(expname,"br.png"))



