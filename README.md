# DeconvDec
Code for the model proposed in our paper *Deconvolution-Based Global Decoding for Neural Machine Translation*, http://aclweb.org/anthology/C18-1276.

## Requirements
* Ubuntu 16.0.4
* Python 3.5
* Pytorch 0.4.1

**************************************************************

## Preprocessing
```
python3 preprocess.py -load_data path_to_data -save_data path_to_store_data 
```
Remember to put the data into a folder and name them *train.src*, *train.tgt*, *valid.src*, *valid.tgt*, *test.src* and *test.tgt*, and make a new folder inside called *data*. For more detailed setting, check the options in the file.

***************************************************************

## Training
```
python3 train.py -log log_name -config config_yaml -gpus id
```
Create your own yaml file for hyperparameter setting.

****************************************************************

## Evaluation
```
python3 train.py -log log_name -config config_yaml -gpus id -restore checkpoint -mode eval
```

*******************************************************************

# Citation
If you use this code for your research, please kindly cite our paper:.
```
@inproceedings{DeconvDec,
  author    = {Junyang Lin and
               Xu Sun and
               Xuancheng Ren and
               Shuming Ma and
               Jinsong Su and
               Qi Su},
  title     = {Deconvolution-Based Global Decoding for Neural Machine Translation},
  booktitle = {Proceedings of the 27th International Conference on Computational
               Linguistics, {COLING} 2018, Santa Fe, New Mexico, USA, August 20-26,
               2018},
  pages     = {3260--3271},
  year      = {2018}
}
```

