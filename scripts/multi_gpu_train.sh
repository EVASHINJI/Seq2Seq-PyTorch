TRAIN_PATH=data/fra2eng/fra_eng.pairs
DEV_PATH=data/fra2eng/fra_eng.dev
SRC_VOCAB=data/fra2eng/src_vocab_file
TGT_VOCAB=data/fra2eng/tgt_vocab_file

# Start training
CUDA_VISIBLE_DEVICES=1,2,3,4 horovodrun -np 4 -H localhost:4 python runModel.py \
         --device 1,2,3,4 \
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