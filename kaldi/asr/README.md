This README provides steps to reproduce the training of baseline HMM-based models using [Kaldi](https://kaldi-asr.org/)

# Prerequisites
* Follow the steps provided [here](https://kaldi-asr.org/doc/install.html) to install Kaldi
* Install SRILM toolkit by running `./install_srilm.sh` from the `kaldi/tools` directory
* Install mp3 handler for sox using `sudo apt install libsox-fmt-mp3`

# Training

## Source kaldi root directory

* Open `path.sh` 
* Change `<kaldi-root-dir>` to the path where you've cloned and installed kaldi

## Data preperation
### Training data:
* Go to `data/train_100h/` directory
* Open `wav.scp`
* Replace `fullpath/to/data` with full path of the directory where `Gramvaani_Train_100` folder is stored
### Test data:
* Go to `data/dev_5h/` directory
* Open `wav.scp`
* Replace `fullpath/to/data` with full path of the directory where `Gramvaani_Dev_5` folder is stored


## Start training and testing
* Use commands `./Run_tdnn_1i.sh` and `./Run_cnn-tdnn.sh` to train TDNN and CNN-TDNN models respectively
* Both scripts start the entire process form the scratch, including language model (LM) preperation, feature extraction, and GMM training
* Therefore, if you have already run one model, then you can skip those initial common stages by setting different switched to zero
* For example, once you have prepared the LM, you can set `lm_prep=0` in run file
