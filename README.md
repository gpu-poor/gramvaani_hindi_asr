# gramvaani_hindi_asr

 Baseline model's performance on Gram vaani 5h dev set

|Model Type|Framework|Hyper-params |Test Set |WER|CER|LM|
|---|---|---|---|---|---|---
Transformer|ESPnet|12 layers; nbpe: 1000|dev_5h| 37.3|19.6|Not used|
|Conformer|ESPnet| 12 layers ; kernel size 15 ; nbpe: 1000|dev_5h|34.8|19|Not used|
|TDNN|Kaldi|13 layers|dev_5h|30.12|16.12|3-gram LM|
|CNN-TDNN|Kaldi|12 layers tdnn with cnn front-end|dev_5h|30.93|17.03|3-gram LM|

# 

| Baseline | Model             | Pretraining data                               | Finetuning data              | Model                                                        | dev WER |
| -------- | ----------------- | ---------------------------------------------- | ---------------------------- | ------------------------------------------------------------ | ------- |
| I        | Wav2Vec 2.0 Base  | 1000 hours Gramvaani unlabelled                | No finetuning                | [download](https://drive.google.com/file/d/1L6wh5hQ_-K0szGy9tpzF-jwm8gSa5pJe/view?usp=sharing) | N.A     |
| II       | Wav2Vec 2.0 Base  | 1000 hours Gramvaani unlabelled                | 100 hours Gramvaani labelled | [download](https://drive.google.com/file/d/1MpUAtQ0TM7Vn92rIOxGUVj1oHOGrVDee/view?usp=sharing) | 35.973  |
| III      | Wav2Vec 2.0 Large | 1000 hours Gramvaani unlabelled                | No finetuning                |                                                              | N.A     |
| IV       | Wav2Vec 2.0 Large | 1000 hours Gramvaani unlabelled                | 100 hours Gramvaani labelled |                                                              |         |
| V        | Wav2Vec 2.0 Base  | N.A (pretrained model from AI4Bharat)          | 100 hours Gramvaani labelled | [download](https://drive.google.com/file/d/1OH-IQIk408wiUgWBoyAzeQEuC9KoXHOG/view?usp=sharing) | 33.307  |
| VI       | Wav2Vec 2.0 Base  | N.A (pretrained model from Open-Speech-EkStep) | 100 hours Gramvaani labelled | [download](https://drive.google.com/file/d/1s9bAhLBOaWpKc2dd1W_fbgVmtwKhfMa-/view?usp=sharing) | 34.328  |

> Clone [this](https://github.com/anish9208/gramvaani_hindi_asr) repository to try recipes and trained baseline models in various framework

> Espnet based Transformer/Conformer recipe and trained baseline models are avilable in `./espnet/asr/` 
> 
> Kaldi based recipes are available in `./kaldi/asr `
> 
> Wav2vec2.0 based models and config files are available at `./wav2vec2.0`
