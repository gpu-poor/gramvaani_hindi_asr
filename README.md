# gramvaani_hindi_asr

 Baseline model's performance on Gram vaani 5h dev set

|Model Type|Framework|Hyper-params |Test Set |WER|CER|
|---|---|---|---|---|---
Transformer|ESPnet|12 layers; nbpe: 1000|dev_5h| 37.3|19.6|
|Conformer|ESPnet| 12 layers ; kernel size 15 ; nbpe: 1000|dev_5h|34.8|19|
|

> Clone [this](https://github.com/anish9208/gramvaani_hindi_asr) repository to try recipes and trained baseline models in various framework

> Espnet based Transformer/Conformer recipe and trained baseline models are avilable in `./espnet/asr/` 

> Wav2Vec based models are coming soon...

> Kaldi based models are coming soon...