TRAIN_PATH=data/fra2eng/fra_eng.pairs
DEV_PATH=data/fra2eng/fra_eng.dev
SRC_VOCAB=data/fra2eng/src_vocab_file
TGT_VOCAB=data/fra2eng/tgt_vocab_file
check=/home/hanxun_zhong/Seq2Seq-PyTorch/experiment/5000.pt
# Start training
CUDA_VISIBLE_DEVICES=4,5,6,7 horovodrun -np 4 -H localhost:4 --timeline-filename timeline.json python runModel.py \
         --train_path $TRAIN_PATH \
         --dev_path $DEV_PATH \
         --device 4,5,6,7 \
         --src_vocab_file $SRC_VOCAB \
         --tgt_vocab_file $TGT_VOCAB \
         --batch_size 32 \
         --bidirectional \
         --use_attn 
        #  --resume  \
        #  --load_checkpoint $check \
        # --phase infer \
        # --beam_width 5 
