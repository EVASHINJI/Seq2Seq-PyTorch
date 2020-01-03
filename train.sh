TRAIN_PATH=data/fra2eng/fra_eng.pairs
DEV_PATH=data/fra2eng/fra_eng.dev
check=~/repos/chatbot/S2S_pytorch/Seq2Seq-PyTorch/experiment/1992.pt
# Start training
python runModel.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --device 0 --load_checkpoint $check --phase infer