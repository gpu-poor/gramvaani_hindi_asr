# Pretrained models

| Baseline | Model             | Pretraining data                               | Finetuning data              | Model                                                        | dev WER |
| -------- | ----------------- | ---------------------------------------------- | ---------------------------- | ------------------------------------------------------------ | ------- |
| I        | Wav2Vec 2.0 Base  | 1000 hours Gramvaani unlabelled                | No finetuning                | [download](https://drive.google.com/file/d/1L6wh5hQ_-K0szGy9tpzF-jwm8gSa5pJe/view?usp=sharing) | N.A     |
| II       | Wav2Vec 2.0 Base  | 1000 hours Gramvaani unlabelled                | 100 hours Gramvaani labelled | [download](https://drive.google.com/file/d/1MpUAtQ0TM7Vn92rIOxGUVj1oHOGrVDee/view?usp=sharing) | 35.973  |
| III      | Wav2Vec 2.0 Large | 1000 hours Gramvaani unlabelled                | No finetuning                | [download](https://drive.google.com/file/d/1VtEWsjZgXnZvaCQj5aDx8t0WlfG1zVez/view?usp=sharing)                                                             | N.A     |
| IV       | Wav2Vec 2.0 Large | 1000 hours Gramvaani unlabelled                | 100 hours Gramvaani labelled |                                                              |         |
| V        | Wav2Vec 2.0 Base  | N.A (pretrained model from AI4Bharat)          | 100 hours Gramvaani labelled | [download](https://drive.google.com/file/d/1OH-IQIk408wiUgWBoyAzeQEuC9KoXHOG/view?usp=sharing) | 33.307  |
| VI       | Wav2Vec 2.0 Base  | N.A (pretrained model from Open-Speech-EkStep) | 100 hours Gramvaani labelled | [download](https://drive.google.com/file/d/1s9bAhLBOaWpKc2dd1W_fbgVmtwKhfMa-/view?usp=sharing) | 34.328  |

All the WERs mentioned in tha above table are obtained by finetuning the pretrained models with CTC after which the CTC model has been evaluated on the development set without the use of any language model.

Baselines V and VI are the result of finetuning the available wav2vec2.0 base models from the [IndicWav2Vec](https://indicnlp.ai4bharat.org/indicwav2vec/) initiative of AI4Bharat and [Open-Speech-EkStep](https://github.com/Open-Speech-EkStep/vakyansh-models) respectively.

[IndicWav2Vec](https://arxiv.org/abs/2111.03945) is a multilingual speech model pretrained on 40 Indian langauges. 

The Open-Speech-Ekstep model used for the Baseline VI is a Vakyansh Open Source model that has been pretrained on 4200 hours of Hindi data.

This readme provides steps to reproduce the training/inference of baseline wav2vec2.0 transformer models using [fairseq](https://github.com/pytorch/fairseq)

# Training a new model with CLI tools

## Installation

Please follow the [requirements and installation](https://github.com/pytorch/fairseq#requirements-and-installation) instructions from the fairseq repository.

In addition install the `soundfile` library:

```sh
pip install soundfile
```

## Preparing data manifest

Gram Vaani data has `.mp3` files with mix of sampling  rates ranging from 8KHz to 48 KHz for both labelled 100 hours of data & unlabelled 1000 hours of data.

All the audio files have been converted to `.wav` format and has been uniformly sampled to 16KHz.

After this step, run:

```sh
python examples/wav2vec/wav2vec_manifest.py /path/to/waves --dest /manifest/path --ext $ext --valid-percent $valid
```

`$ext` should be set to flac, wav, or whatever format your dataset happens to use that soundfile can read. In our case, since we converted the data to `.wav` format, `$ext` was replaced with `wav` in the above command.

`$valid` should be set to some reasonable percentage (like 0.01) of training data to use for validation.

Upon execution, `train.tsv` and `valid.tsv` would be generated in the destination directory.

## Training a wav2vec2.0 model

Input is expected to be single channel, sampled at 16kHz.

```sh
fairseq-hydra-train \
    task.data=/path/to/data \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/pretraining \
    --config-name config_name
```

If you're training a wav2vec2.0 base model, pass `wav2vec2_base` in place of `config_name` in the above command.

If you're training a wav2vec2.0 large model, pass `wav2vec2_large` in place of `config_name` in the above command.

Note: you can simulate 64 GPUs by using k GPUs and adding command line parameters (before `--config-dir`) `distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'`. Here, when training a wav2vec2.0 base model, x = 64/k. However, while training a wav2vec2.0 large model, x = 128/k.

## Finetune a pre-trained model with CTC

Fine-tuning a model requires parallel audio and labels file, as well as a vocabulary file in fairseq format. A letter vocabulary can be downloaded [here](https://dl.fbaipublicfiles.com/fairseq/wav2vec/dict.ltr.txt). However, this is the vocabulary for the LibriSpeech data. You would need to generate a similar vocabulary file with the letters and the corresponding number of occurances of that letter in the text, from the text file provided in the dataset. An example script that generates labels for the Librispeech dataset from the tsv file produced by wav2vec_manifest.py can be used as follows:

```sh
python libri_labels.py /path/to/tsv --output-dir /output/dir --output-name $split
```

In place of `$split` pass `train` while generating the finetuning data for the training data. Similarly, while generating the finetuning data for the validation set, pass `valid` or `dev` and accordingly modify the `valid_subset` variable in the finetuning config file.

**NOTE:** This script is written to facilitate the data preparation of the LibriSpeech dataset. Before going ahead with the manifest file preparation, ensure that you modify this script to take the transcriptions from the text file provided in the dataset. The vocabulary that has earlier been generated must be named as `dict.ltr.txt` and placed in the same directory as the rest of the finetuning data files.

Finetuning a wav2vec2.0 model would then be done upon the succesfull execution of the following command:

```sh
fairseq-hydra-train \
    task.data=/path/to/data \
    model.w2v_path=/path/to/model.pt \
    --config-dir /path/to/fairseq-py/examples/wav2vec/config/finetuning \
    --config-name config_name
```

There are other config files in the config/finetuning directory that can be used to fine-tune on different models. You can specify the right config via the `--config-name` parameter.

Note: you can simulate 24 GPUs by using k GPUs and adding command line parameters (before `--config-dir`) `distributed_training.distributed_world_size=k` `+optimization.update_freq='[x]'` where x = 24/k.

## Evaluating a CTC model

Evaluating a CTC model with a language model requires [flashlight python bindings](https://github.com/facebookresearch/flashlight/tree/master/bindings/python) (previously called [wav2letter](https://github.com/facebookresearch/wav2letter) to be installed.

Data preparation process for the test data is same as the data preparation process followed in the finetuning stage.

To decode without any language model, proceed with the execution of the following command:

```sh
python examples/speech_recognition/infer.py /path/to/test/data --task audio_finetuning \
--nbest 1 --path /path/to/model --gen-subset $subset --results-path /path/to/save/results/for/sclite --criterion ctc --labels ltr --max-tokens 4000000 --post-process letter
```

