DEVICES="1,2,3,4"
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
         --use_attn
        #  --resume  \
        #  --load_checkpoint $check \
        # --phase infer \
        # --beam_width 5 
