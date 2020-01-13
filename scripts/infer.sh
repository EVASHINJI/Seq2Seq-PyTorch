DEVICES="cpu"

TRAIN_PATH=data/fra2eng/fra_eng.pairs
DEV_PATH=data/fra2eng/fra_eng.dev
SRC_VOCAB=data/fra2eng/src_vocab_file
TGT_VOCAB=data/fra2eng/tgt_vocab_file
checkpoint="experiment/4000.pt"

# Start inference
python runModel.py \
        --device $DEVICES \
        --train_path $TRAIN_PATH \
        --dev_path $DEV_PATH \
        --src_vocab_file $SRC_VOCAB \
        --tgt_vocab_file $TGT_VOCAB \
        --bidirectional \
        --use_attn \
        --load_checkpoint $checkpoint \
        --phase infer \
        --beam_width 5 
