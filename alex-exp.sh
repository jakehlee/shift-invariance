python extraction/ex-alexnet-AA.py datasets/COCO_small_white extraction/features/small_white_alexnet_AA;
python extraction/ex-alexnet-AA.py datasets/COCO_large_white extraction/features/large_white_alexnet_AA;
python extraction/ex-alexnet.py datasets/COCO_small_white extraction/features/small_white_alexnet;
python extraction/ex-alexnet.py datasets/COCO_large_white extraction/features/large_white_alexnet;

python extraction/cosine_calc.py extraction/features/small_white_alexnet_AA;
python extraction/cosine_calc.py extraction/features/large_white_alexnet_AA;
python extraction/cosine_calc.py extraction/features/small_white_alexnet;
python extraction/cosine_calc.py extraction/features/large_white_alexnet;

