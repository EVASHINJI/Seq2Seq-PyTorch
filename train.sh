TRAIN_PATH=data/toy_reverse/train/data.txt
DEV_PATH=data/toy_reverse/dev/data.txt
# Start training
python runModel.py --train_path $TRAIN_PATH --dev_path $DEV_PATH --device 0