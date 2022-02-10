#!/bin/bash
# Set bash to 'debug' mode, it will exit on :
# -e 'error', -u 'undefined variable', -o ... 'error in pipeline', -x 'print commands',
set -e
set -u
set -o pipefail

train_set="Gramvaani_Train_100" 
valid_set="Gramvaani_Dev_5" 
test_sets="Gramvaani_Dev_5"

#asr_config=conf/train_asr_rnn.yaml
# asr_config=conf_Conformer/train_asr_transformer.yaml 
asr_config=conf_Conformer/train_asr_confformer.yaml
inference_config=conf/decode_asr.yaml

# FIXME(kamo):
# The results with norm_vars=True is odd.
# I'm not sure this is due to bug.

./asr.sh \
    --nbpe 1000 \
    --stage 11 \
    --stop_stage 12 \
    --ngpu 1 \
    --lang "${lang}" \
    --dumpdir "/nlsasfs/home/nltm-pilot/vasistal/ASR_project/scripts_from_nithya/graamvani_data_expts/dump" \
    --local_data_opts "--lang ${lang}" \
    --use_lm false \
    --gpu_inference true \
    --token_type bpe \
    --feats_type fbank_pitch \
    --asr_args "--normalize_conf norm_vars=False " \
    --asr_config "${asr_config}" \
    --inference_config "${inference_config}" \
    --train_set "${train_set}" \
    --valid_set "${valid_set}" \
    --test_sets "${test_sets}" \
    --srctexts "data/${train_set}/text" "$@"
