import argparse

parser = argparse.ArgumentParser()

# Model
parser.add_argument('--max_steps', action='store', dest='max_steps', 
                    help='Num steps to train', default=500000, type=int)
parser.add_argument('--max_epochs', action='store', dest='max_epochs', 
                    help='Num epochs to train', default=15, type=int)
parser.add_argument('--skip_steps', action='store', dest='skip_steps', 
                    help='Skip steps to train', default=0, type=int)
parser.add_argument('--checkpoint_every', action='store', dest='checkpoint_every', 
                    help='Num batches to checkpoint', default=500, type=int)
parser.add_argument('--print_every', action='store', dest='print_every', 
                    help='Num batches to print', default=50, type=int)
parser.add_argument('--embedding_size', action='store', dest='embedding_size', 
                    help='Size of embedding', default=100, type=int)
parser.add_argument('--hidden_size', action='store', dest='hidden_size', 
                    help='Size of each RNN hidden layer', default=128, type=int)
parser.add_argument('--bidirectional', action='store_true', dest='bidirectional', 
                    help='bidirectional RNN', default=False)
parser.add_argument('--batch_size', action='store', dest='batch_size', 
                    help='Size of batch', default=32, type=int)
parser.add_argument('--beam_width', action='store', dest='beam_width', 
                    help='Beam width when using beam search decoder.', default=1, type=int)
parser.add_argument('--init_weight', action='store', dest='init_weight', 
                    help='Initial weights from [-this, this]', default=0.08, type=float)
parser.add_argument('--clip_grad', action='store', dest='clip_grad', 
                    help='Clip gradients to this norm', default=5.0, type=float)
parser.add_argument('--learning_rate', action='store', dest='learning_rate', 
                    help='Learning rate', default=0.001, type=float)
parser.add_argument('--best_ppl', dest='best_ppl', 
                    help='best ppl to save model.', default=100000.0, type=float)
parser.add_argument('--use_attn', action='store_true', dest='use_attn', 
                    help='If use attention', default=False)
parser.add_argument('--max_src_length', action='store', dest='max_src_length', 
                    help='max length of source', default=50, type=int)
parser.add_argument('--max_tgt_length', action='store', dest='max_tgt_length', 
                    help='max length of target', default=50, type=int)
parser.add_argument('--teacher_forcing_ratio', action='store', dest='teacher_forcing_ratio', 
                    help='teacher forcing ratio', default=0.5, type=float)
# parser.add_argument('--', action='store', dest='', 
                    # help='', default=)
# Files
parser.add_argument('--train_path', action='store', dest='train_path', help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path', help='Path to dev data')
parser.add_argument('--test_path', action='store', dest='test_path', help='Path to test data')

parser.add_argument('--src_vocab_file', action='store', dest='src_vocab_file', 
                    help='Path to source vocab')
parser.add_argument('--tgt_vocab_file', action='store', dest='tgt_vocab_file', 
                    help='Path to target vocab')
parser.add_argument('--src_vocab_size', action='store', dest='src_vocab_size', 
                    help='Size of source vocab', default=40000, type=int)
parser.add_argument('--tgt_vocab_size', action='store', dest='tgt_vocab_size', 
                    help='Size of target vocab', default=40000, type=int)

parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The file path of the checkpoint to load.')

parser.add_argument('--model_dir', action='store', dest='model_dir', default='./experiment', 
                    help='Path to model directory.')
parser.add_argument('--best_model_dir', action='store', dest='best_model_dir', default='./experiment/best', 
                    help='Path to best model directory.')
parser.add_argument('--max_checkpoints_num', action='store', dest='max_checkpoints_num', default=5, 
                    help='Max num of checkpoints', type=int)

parser.add_argument('--resume', action='store_true', dest='resume', default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint. If load_checkpoint is set, then train from loaded.')
parser.add_argument('--log_level', action='store', dest='log_level', default='info', help='Logging level.')
parser.add_argument('--log_file', action='store', dest='log_file', default='info', help='Logging file path.')
parser.add_argument('--device', action='store', dest='device', default=None, help='GPU device.', type=str)
parser.add_argument('--phase', action='store', dest='phase', default='train', help='train or infer')

opt = parser.parse_args()