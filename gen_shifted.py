import sys, os, ast, time
import numpy as np
import matplotlib.pyplot as plt
import cv2
import yaml

from pycocotools.coco import COCO


def usage():
    print("Usage: python gen_shifted.py shift_config")
    sys.exit(0)

def crop_obj(img, mask, bbox):
    x, y, w, h = [round(x) for x in bbox]
    img_cropped = np.array(img)[y:y+h,x:x+h]
    mask_cropped = np.array(mask)[y:y+h,x:x+h]
    mask_cropped = mask_cropped[:,:,np.newaxis]
    
    # stack to BGRA
    bgra = np.concatenate([img_cropped, mask_cropped], axis=2)
    return bgra

def shift_patch(patch, bg, patchsize, bgsize, outdir, imgId, annId, expname, stride=1):
    
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    
    currdir = os.path.join(outdir, str(imgId)+"-"+str(annId)) 
    os.mkdir(currdir)
    currdir = os.path.join(currdir, 'sub')
    os.mkdir(currdir)

    
    patch = cv2.resize(patch, patchsize)
    bg = cv2.resize(bg, bgsize)

    a, b, _ = patch.shape
    m, n, _ = bg.shape

    counter = 0
    for i in range(0, m-a+1, stride):
        for j in range(0, n-b+1, stride):
            out = bg.copy()

            p_alpha = patch[:,:,3]
            out[i:i+a,j:j+b,0] = (1 - p_alpha) * bg[i:i+a,j:j+b,0] + \
                p_alpha * patch[:,:,0]
            out[i:i+a,j:j+b,1] = (1 - p_alpha) * bg[i:i+a,j:j+b,1] + \
                p_alpha * patch[:,:,1]
            out[i:i+a,j:j+b,2] = (1 - p_alpha) * bg[i:i+a,j:j+b,2] + \
                p_alpha * patch[:,:,2]

            out_name = "{}_{}_{}.png".format(i, j, expname)
            out_path = os.path.join(currdir, out_name)

            cv2.imwrite(out_path, out)
            counter += 1

    return counter

if __name__ == "__main__":

    if len(sys.argv) != 2:
        usage()

    # Load config
    with open(sys.argv[1], 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    coco_cats = config['coco_categories']
    coco_imgpercat = config['coco_imgpercat']
    bg_path = config['bg_path']
    bg_size = ast.literal_eval(config['bg_size'])
    patch_size = ast.literal_eval(config['patch_size'])
    out_dir = config['out_dir']
    exp_name = config['exp_name']

    # Load background image
    bg = cv2.imread(bg_path, cv2.IMREAD_COLOR)

    # Load COCO
    dataDir = 'coco/'
    dataType = 'train2017'
    annFile = '{}/annotations/instances_{}.json'.format(dataDir,dataType)
    coco = COCO(annFile)

    # Time measure
    start = time.time()

    # For each categrory
    for cat in coco_cats:
        # Get Category ID
        catId = coco.getCatIds(catNms=cat)
        # Get Images with these categories
        imgIds = sorted(coco.getImgIds(catIds=catId))
        counter = 0
        for imgId in imgIds:
            if counter == coco_imgpercat:
                break
            img = coco.loadImgs(imgId)[0]
            filename = img['file_name']
            I = cv2.imread(os.path.join(dataDir, dataType, filename), cv2.IMREAD_COLOR)
            annIds = coco.getAnnIds(imgIds=img['id'], catIds=catId, iscrowd=None)
            ann = coco.loadAnns(annIds[0])[0]
            mask = coco.annToMask(ann)
            bbox = ann['bbox']
            
            # If labeled image is too small, don't use it
            if bbox[2] < 75/2 and bbox[3] < 75/2:
                continue

            patch = crop_obj(I, mask, bbox)
           
            ret = shift_patch(patch, bg, patch_size, bg_size, out_dir, img['id'], ann['id'], exp_name)
            
            counter += 1

            print("Wrote", ret, "images")

    end = time.time()

    print("Time Elapsed:", end-start)
