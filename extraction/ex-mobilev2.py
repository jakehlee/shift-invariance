import sys, os
import numpy as np
import csv
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.autograd import Variable

BATCH = 128

def usage():
    print("Usage: python ex-model.py dataset_dir out_dir")
    sys.exit(0)

if __name__ == "__main__":

    if len(sys.argv) != 3:
        usage()

    img_dir = sys.argv[1]
    out_dir = sys.argv[2]

    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    img_transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])])

    listdir = os.listdir(img_dir)
    print(listdir)
    subdirs = []
    for d in listdir:
        if os.path.isdir(os.path.join(img_dir, d)):
            subdirs.append(d)
    
    model = models.mobilenet_v2(pretrained=True)
    model.cuda()
    model.eval()

    for d in subdirs:
        img_subdir = os.path.join(img_dir, d)
        out_subdir = os.path.join(out_dir, d)
        os.mkdir(out_subdir)

        dataset = datasets.ImageFolder(root=img_subdir, transform=img_transform)
        dataset_loader = data.DataLoader(dataset, batch_size=BATCH)

        fc_out = []
        class_out = []
        for i, (img_batch, _) in enumerate(dataset_loader,0):
            print("Extracting batch {}".format(i))
            img_batch = img_batch.cuda()
            ptr = i * BATCH

            fc_buf = model(img_batch)
            class_buf = F.softmax(fc_buf, dim=1).tolist()
            fc_buf = fc_buf.tolist()

            for j in range(len(fc_buf)):
                img_name = os.path.split(dataset.imgs[ptr+j][0])[-1]
                fc_out.append([img_name] + fc_buf[j])

        with open(os.path.join(out_subdir, "fc.csv"), 'w') as f:
            wr = csv.writer(f)
            wr.writerows(fc_out)

