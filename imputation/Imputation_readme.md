# Imputation of Sentinel-1 and Sentinel-2 images using U-Net
This repository contains code for training a U-Net model to impute missing pixels in Sentinel-1 and Sentinel-2 images using MODIS data as a reference. The model is trained on cropped geotiffs of size 64x64 pixels.

## Pre-requisites

1. Cropped Images (MODIS, Sent-1 and Sent-2) of size 64x64 available in a directory.
2. Install pytorch lightning and other dependencies.

## Training the U-Net model for imputation
1. Run the following command to train the U-Net model for imputation:
```bash
python3 imputation/imputation_model.py
```
2. The model will be saved in the `imputation/models` directory.

## Perform imputation
1. Run the following command to perform imputation:
```bash
python3 imputation/perform_imputation.py        
```
2. The imputed images will be saved in the `imputation/imputed_s2` directory for Sentinel 2.