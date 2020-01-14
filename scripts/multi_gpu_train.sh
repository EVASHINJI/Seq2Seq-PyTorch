DEVICES="0,1,2,3"
DEVICES_NUM=4

TRAIN_PATH=data/fra2eng/fra_eng.pairs
DEV_PATH=data/fra2eng/fra_eng.dev
SRC_VOCAB=data/fra2eng/src_vocab_file
TGT_VOCAB=data/fra2eng/tgt_vocab_file

# Start training
CUDA_VISIBLE_DEVICES=$DEVICES horovodrun -np $DEVICES_NUM -H localhost:$DEVICES_NUM python runModel.py \
        --device $DEVICES \
        --train_path $TRAIN_PATH \
        --dev_path $DEV_PATH \
        --src_vocab_file $SRC_VOCAB \
        --tgt_vocab_file $TGT_VOCAB \
        --bidirectional \
        --use_attn \
        --random_seed 2808 \
        --model_dir experiment/multi_gpu \
        --best_model_dir experiment/multi_gpu/best \
        --resume
