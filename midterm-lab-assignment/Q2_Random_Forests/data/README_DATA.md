# Q2 data folder

## Heart Disease

Download: https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci

Copy the main CSV to `data/heart.csv`.

## Intel Image Classification

Download: https://www.kaggle.com/datasets/puneet6060/intel-image-classification

Extract the training folder so that its class folders are found below `data/intel/seg_train/`. One accepted layout is:

```text
data/intel/seg_train/
  buildings/
  forest/
  glacier/
  mountain/
  sea/
  street/
```

If the archive contains `seg_train/seg_train/buildings/...`, you can copy the inner `seg_train` directory or let the script search for the nested class root.

The script builds a stratified 80/20 split from the six labelled training folders. It does not use the unlabeled `seg_pred` directory.
