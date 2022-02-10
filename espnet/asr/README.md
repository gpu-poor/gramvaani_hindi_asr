
This readme provides steps to reproduce the training/inference of baseline trasformer/conformer models using [espnet](https://github.com/espnet/espnet)


# ESPnet installation
* please follow the steps provided [here](https://espnet.github.io/espnet/installation.html#step-2-installation-espnet)

# Training

## source espnet environment

* open `path.sh` 
* change `<espnet-clone>` to the path where you've cloned and installed espnet
* run the command > `source path.sh`

* this should change your working env to espnet env. To make sure try `which python` command on shell and you should see  `<espnet-clone>/espnet/espnet-master/tools/venv/bin/python` on your shell prompt. 

## Setup training data

* put directory `Gramvaani_Dev_5` and `Gramvaani_Train_100` inside `data` directory 
* if you don't have  `data` dir then create one with this command > `makdir ./data`
* make sure that you have `wav.scp, segments, text, utt2spk, spk2utt` files inside both of above said directories
* make sure in `run.sh`  the variables `train_set, valid_set, test_set` are assigned appropriate directory path. It is assumed that these path will be inside `./data` dir
* they should look like
 ```bash
train_set="Gramvaani_Train_100" 
valid_set="Gramvaani_Dev_5" 
test_sets="Gramvaani_Dev_5"
```

## Transformer or Conformer?

* The model for which you want to run training or decodeing/inference should be specified using the config.
* in `run.sh` the variable `asr_config` points to the yaml config file for your model.
* transformer model config : `conf_Conformer/train_asr_transformer.yaml`
* conformer model config : `conf_Conformer/train_asr_confformer.yaml`
* the `conf/decode_asr.yaml` file contains genric inference parameters like
```yaml
lm_weight: 0.2
ctc_weight: 0.3
beam_size: 20
```

## Start training

* use the commands shown below
```bash
 . path.sh # source the espnet env 
 ./run.sh --stage 3 --stop-stage 5  # data prep, tokeniser training
 CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 9 --stop-stage 10 --ngpu 4 # for a machine where 4 gpus are avilable. you can change CUDA_VISIBLE_DEVICES and --ngpu as per your own setup
```
* the trained model can be found in `./exp/asr_tain.....` dir. the best model is choiced based on validation accuracy and best model file is named  `valid.acc.best.pth`

# Eval on validation or test set

* The dataset on for eval is defined by `test_sets` variable in  `run.sh`
* To start eval use command below
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 11 --stop-stage 12 --ngpu 4 # stage 11 to 12 for decoding/eval
```

# Eval on validation or test set using pre-trained baseline model
> [Transformer trained model .pth](https://zenodo.org/record/6033179/files/valid.acc.best.pth?download=1)

> [Conformer trained model .pth](https://zenodo.org/record/6037023/files/valid.acc.best.cf.pth?download=1)


### Transformer
*******
```bash
. path.sh

# make required dir if not already there

mkdir -p ./exp/asr_train_asr_transformer_raw_bpe_normalize_confnorm_varsFalse/

mkdir -p ./data/token_list/bpe_unigram1000/

mkdir -p ./exp/asr_stats_fbank_pitch/train/

#download pytorch model
wget https://zenodo.org/record/6033179/files/valid.acc.best.pth?download=1 -P ./exp/asr_train_asr_transformer_raw_bpe_normalize_confnorm_varsFalse/valid.acc.best.pth

#copy required config 
cp transformer_files/config.yaml ./exp/asr_train_asr_transformer_raw_bpe_normalize_confnorm_varsFalse/

#copy tokeniser model
cp ./transformer_files/bpe.model ./data/token_list/bpe_unigram1000/

#copy stats
cp  ./transformer_files/feats_stats.npz  ./exp/asr_stats_fbank_pitch/train/

cp ./transformer_files/run.sh run.sh

#run stage 11-12
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 11 --stop-stage 12 --ngpu 4

```

### Conformer
********
```bash
. path.sh
# make required dir if not already there

mkdir -p ./exp/asr_train_asr_confformer_fbank_pitch_bpe_normalize_confnorm_varsFalse

mkdir -p ./data/token_list/bpe_unigram1000/

mkdir -p ./exp/asr_stats_fbank_pitch/train/

#download pytorch model
wget https://zenodo.org/record/6037023/files/valid.acc.best.cf.pth?download=1 -P ./exp/asr_train_asr_confformer_fbank_pitch_bpe_normalize_confnorm_varsFalse/valid.acc.best.pth

#copy required config 
cp conformer_files/config.yaml ./exp/asr_train_asr_confformer_fbank_pitch_bpe_normalize_confnorm_varsFalse/

#copy tokeniser model
cp ./conformer_files/bpe.model ./data/token_list/bpe_unigram1000/

#copy stats
cp  ./conformer_files/feats_stats.npz  ./exp/asr_stats_fbank_pitch/train/

cp ./conformer_files/run.sh run.sh

#run stage 11-12
CUDA_VISIBLE_DEVICES=0,1,2,3 ./run.sh --stage 11 --stop-stage 12 --ngpu 4
```