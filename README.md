# Seq2Seq-PyTorch

基于[IBM/pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq)进行简化与拓展，在[PyTorch](http://pytorch.org/) 1.3.1上实现的Sequence-to-Sequence (Seq2Seq)模型。支持Attention机制，Beam Search，多卡训练等。

This is a framework for Sequence-to-Sequence (Seq2Seq) models implemented in [PyTorch](http://pytorch.org/) 1.3.1. And this is simplified and expanded on the basis of [IBM/pytorch-seq2seq](https://github.com/IBM/pytorch-seq2seq), implemented Attention mechanism, Beam Search, Multi-GPU training, etc.



### 依赖包 / Requirements
运行程序所需依赖包建议版本。

In order to run the program, here are a list of packages with the suggested versions.

```
- python   3.6
- pytorch  1.3.1
- horovod  0.18.2
- numpy
- tqdm
```



### 数据 / Dataset
PyTorch官网Seq2Seq教程中的英法翻译数据。已转换为所需格式放至“data/fra2eng”目录。原始数据。

English-French translation data in the PyTorch official Seq2Seq tutorial. It has been converted into the required format in “data/fra2eng” directory. Raw data.



### 快速开始 / Quick Start
#### 训练 / Train
```
# Train on single-GPU
bash scripts/train.sh 

# Train on multi-GPU
bash scripts/multi_gpu_train.sh
```

#### 预测 / Inference 
```
bash scripts/infer.sh
```


### 参数设置 / Configuration Parameters
```
*[files] // file paths
-load_checkpoint       : File path of the checkpoint to load. //需要恢复的检查点路径。
-train_path            : Train data file path. //训练集路径。
-dev_path              : Dev data file path. //验证集路径。
-test_path             : Train data file path. //测试集路径。
-src_vocab_file        : Source vocab file path. //输入文件词表路径。
-tgt_vocab_file        : Target vocab file path. //输出文件词表路径。
-model_dir             : Path to model directory. //模型保存目录。
-best_model_dir        : Path to best model directory. //最好模型保存目录。
-max_checkpoints_num   : Max num of checkpoints. //最多保存模型数量。
-log_level             : Logging level. //日志的输出等级。
-log_file              : Logging file path. //日志的输出路径。

*[learn] // hyperparamters for model learning 
-batch_size            : Size of batch. //batch大小。
-resume                : Indicates if training has to be resumed from the latest checkpoint. If load_checkpoint is set, then train from loaded. //是否从最新的检查点恢复训练，若指定load_checkpoint，则从指定检查点恢复。
-max_steps             : Maximum num of steps for training. //最大训练步数。
-max_epochs            : Maximum num of epochs for training. //最大训练轮数。
-skip_steps            : Num of steps skipped at the beginning of training. //在训练开始时跳过步数量。
-checkpoint_every      : Num of batches to checkpoint. //每多少步保存检查点。
-print_every           : Num of batches to print loss. //每多少步输出loss。
-init_weight           : Initial weights from [-this, this]. //参数初始化范围。
-clip_grad             : Clip gradients to this norm. //最大梯度截断。
-learning_rate         : Learning rate. //学习率
-best_ppl              : Initial ppl threshold for saving best model. //用做保存最好模型的初始PPL阈值。

*[structure] // hyperparamters for model structure
-src_vocab_size        : Size of source vocab. //输入词表大小。
-tgt_vocab_size        : Size of target vocab. //输出词表大小。
-embedding_size        : Size of embedding. //词向量维度。
-rnn_cell              : Type of RNN cell. gru or lstm. //RNN的类型，gru或lstm。
-n_hidden_layer        : Num of hidden layer in each RNN. //RNN的隐藏层数。
-hidden_size           : Size of each RNN hidden layer. //RNN的隐藏层维度。
-bidirectional         : If use bidirectional RNN. //是否使用双向RNN。
-max_src_length        : Max length of source. //输入的最大长度。
-max_tgt_length        : Max length of target. //输出的最大长度。
-use_attn              : If use attention. //是否用注意力机制。
-teacher_forcing_ratio : teacher forcing ratio. //teacher forcing率。

*[others] // Others
-device                : GPU device. //使用的GPU编号。
-phase                 : train or infer. //训练或预测。
-beam_width            : Beam width when using beam search decoder in inference. //Beam Search宽度。
```



