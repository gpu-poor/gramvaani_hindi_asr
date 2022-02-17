# gramvaani_hindi_asr

 Baseline model's performance on Gram vaani 5h dev set

|Model Type|Framework|Hyper-params |Test Set |WER|CER|LM|
|---|---|---|---|---|---|---
Transformer|ESPnet|12 layers; nbpe: 1000|dev_5h| 37.3|19.6|Not used|
|Conformer|ESPnet| 12 layers ; kernel size 15 ; nbpe: 1000|dev_5h|34.8|19|Not used|
|TDNN|Kaldi|13 layers|dev_5h|31.39|17.19|3-gram LM|
|CNN-TDNN|Kaldi|12 layers tdnn with cnn front-end|dev_5h|32.12|18.03|3-gram LM|

> Clone [this](https://github.com/anish9208/gramvaani_hindi_asr) repository to try recipes and trained baseline models in various framework

> Espnet based Transformer/Conformer recipe and trained baseline models are avilable in `./espnet/asr/` 

> Wav2Vec based models are coming soon...
