# HSDSSM

The implementation of the paper: ""

This project is forked from [TRQ3DNet](https://github.com/LiPang/TRQ3DNet).


**The HSDSSM Codes have not yet been added. It will be added soon.**



## Getting Started

### 1. Preparing your training/testing datasets
Here, we give an example of the single-scale data preparation for ICVL.

Firstly, download ICVL hyperspectral images into ```data/ICVL/filefolder``` folder and 
make sure your initial data are in the following structure. The train-validation-test 
split can be found in ```train_fns.txt``` , ```validation_fns_*.txt``` and 
```test_fns_*.txt```. (Note, we split the 101 testing data into two parts for Gaussian 
and complex denoising, respectively.) You can create the split by yourself or by the function 
```creat_train_val_test``` in ```hsi_data.py```.


```angular2html
data/ICVL
├── filefolder
│    └──ICVL images (*.mat)
├── train_fns.txt
├── validation_fns_gauss.txt
├── validation_fns_complex.txt
├── test_fns_gauss.txt
├── test_fns_complex.txt
```

Modify and run the python code ```hsi_data.py``` to split the data and create training set. 
Then you will get data in the following structure.

```angular2html
data/ICVL
├── filefolder
│    ├──ICVL images (*.mat)
│    ├── trainset
│    │    └──*.mat
│    ├── valset_gauss
│    │    └──*.mat
│    ├── valset_complex
│    │    └──*.mat
│    ├── testset_gauss
│    │    └──*.mat
│    └── testset_complex
│         └──*.mat
├── trainset
│    └──ICVL64_31.npz
```

Modify and run the Matlab code ```Matlab/HSIValData.m``` and ```Matlab/HSITestData.m``` to create validation and testing sets.
Then you will get data in the following structure.

```angular2html
data/ICVL
├── filefolder
├── trainset
│    └──ICVL64_31.npz
├── valset_gauss
│    └──ICVL_512_30
│        └──*.mat
│    └──ICVL_512_50
│        └──*.mat
├── valset_complex
│    └──ICVL_512_noniid
│        └──*.mat
│    └──ICVL_512_complex
│        └──*.mat
├── testset_gauss
│    └──ICVL_512_30
│        └──*.mat
│    └──ICVL_512_50
│        └──*.mat
│    └──ICVL_512_70
│        └──*.mat
│    └──ICVL_512_blind
│        └──*.mat
├── testset_complex
│    └──ICVL_512_noniid
│        └──*.mat
│    └──ICVL_512_stripe
│        └──*.mat
│    └──ICVL_512_deadline
│        └──*.mat
│    └──ICVL_512_impulse
│        └──*.mat
│    └──ICVL_512_mixture
│        └──*.mat
```


### 2. Testing with pretrained models
Read and modify ```basefolder``` and ```savedir``` in ```hsi_test.py```.

* [Blind Gaussian noise removal]:   
```python hsi_test.py -a hsdssm -p gauss -r -rp ./checkpoints/hsdssm/gauss/model_epoch_<epoch_id>.pth --gpu-ids 0```

* [Mixture noise removal]:  
```python hsi_test.py -a hsdssm -p complex -r -rp ./checkpoints/hsdssm/complex/model_epoch_<epoch_id>.pth --gpu-ids 0```

The result can be found in ```result/hsdssm/gauss/res_model_epoch_<epoch_id>/result_ICVL_512.csv``` and ```result/hsdssm/complex/res_model_epoch_<epoch_id>/result_ICVL_512.csv```.

### 3. Training from scratch
* Training a blind Gaussian model firstly by  
```python hsi_denoising_gauss.py -a hsdssm -p gauss -d ./data/ICVL/trainset/ICVL64_31.npz -v ./data/ICVL/valset_gauss --gpu-ids 0```

* Using the pretrained Gaussian model as initialization to train a complex model:  
```python hsi_denoising_complex.py -a hsdssm -p complex -d ./data/ICVL/trainset/ICVL64_31.npz -v ./data/ICVL/valset_complex -r -rp checkpoints/hsdssm/gauss/model_epoch_<epoch_id>.pth --gpu-ids 0```


## Acknowledgement

This repo is mainly based on [TRQ3DNet](https://github.com/LiPang/TRQ3DNet). We thank them immensely for their contributions to the community.
