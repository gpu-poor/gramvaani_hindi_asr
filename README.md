# gramvaani_hindi_asr

This repo contains the baseline model recipes and pre-trained model for GramVanni hindi ASR challenge

|Model Type|Framework|Hyper-params |Test Set |WER|CER|
|---|---|---|---|---|---
Transformer|ESPnet| enc : 12 layer, dec : 6 layers, 8 attn head, 1000 bpe|dev_5h|39|39|
|Conformer|ESPnet| enc : 12 layer, dec : 6 layers, 8 attn head, 1000 bpe|dev_5h|37|37|
|


Espnet based Transformer/Conformer recipe and trained baseline models are avilable in `./espnet/asr/`

Wav2Vec based models are coming soon...