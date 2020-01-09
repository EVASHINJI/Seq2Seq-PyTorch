TRAIN_PATH=data/fra2eng/fra_eng.pairs
DEV_PATH=data/fra2eng/fra_eng.dev
SRC_VOCAB=data/fra2eng/src_vocab_file
TGT_VOCAB=data/fra2eng/tgt_vocab_file
check=~/repos/chatbot/S2S_pytorch/Seq2Seq-PyTorch/experiment/3000.pt
# Start training
python runModel.py \
         --train_path $TRAIN_PATH \
         --dev_path $DEV_PATH \
         --device 0 \
         --src_vocab_file $SRC_VOCAB \
         --tgt_vocab_file $TGT_VOCAB \
         --batch_size 32 \
         --bidirectional \
         --use_attn \
         --resume \
         --load_checkpoint $check \
         --phase infer \
         --beam_width 5 \
