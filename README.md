# OpenNMT-Livebot
This is a re-implementation of the work [Livebot](https://arxiv.org/abs/1809.04938). For reference the originally implementation is [here](https://github.com/lancopku/livebot).

The details of this re-implementation are reported in the paper: [Response to LiveBot: Generating Live Video Comments Based on Visual and Textual Contexts](https://arxiv.org/abs/2006.03022) 
## Requirements
* Python >= 3.6
* Pytorch 1.3.1
* torchvision 0.4.2

## Download source 
Download text & visual data through Google Drive [Link](https://drive.google.com/open?id=1oKyIg_UEyhzsptj4lJ8G1nI-fk7ZFlS_).
```bash 
unzip livebot_source.zip
```
Alternatively you can choose to download the raw dataset though the link in the original github [repo](https://github.com/lancopku/livebot). 

## Data preprocessing
```bash 
cd preprocess_livebot
python process.py
cd ../
python preprocess.py -train_src onmt_data/train_src.txt -train_tgt onmt_data/train_tgt.txt -valid_src onmt_data/valid_src.txt -valid_tgt onmt_data/valid_tgt.txt -save_data onmt_data/data
python preprocess.py -train_src onmt_data/train_src.txt -train_tgt onmt_data/train_tgt.txt -valid_src onmt_data/test/test_src.txt -valid_tgt onmt_data/test/test_tgt.txt -save_data onmt_data/test/data
```

## Train
```bash 
python train.py -mode train -visual_path visual/resnet18.pkl -data onmt_data/data -position_encoding -param_init_glorot -world_size 1 -gpu_ranks 0
```
Note:
- Multiple GPU training current not supported.

## Test
```bash 
#manually set the $best_model accordingly.
python train.py -mode test -test_source onmt_data/test/test-candidate.json -visual_path visual/resnet18.pkl -data onmt_data/test/data -train_from $best_model -valid_batch_size 100 -world_size 1 -gpu_ranks 0
```
