import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', action='store', dest='train_path', help='Path to train data')
parser.add_argument('--dev_path', action='store', dest='dev_path', help='Path to dev data')
parser.add_argument('--expt_dir', action='store', dest='expt_dir', default='./experiment',
                    help='Path to experiment directory. If load_checkpoint is True, then path to checkpoint directory has to be provided')
parser.add_argument('--load_checkpoint', action='store', dest='load_checkpoint',
                    help='The name of the checkpoint to load, usually an encoded time string')
parser.add_argument('--resume', action='store_true', dest='resume', default=False,
                    help='Indicates if training has to be resumed from the latest checkpoint')
parser.add_argument('--log-level', dest='log_level', default='info', help='Logging level.')
parser.add_argument('--device', dest='device', default='info', help='GPU device.')
parser.add_argument('--phase', dest='phase', default='train', help='train or infer')

opt = parser.parse_args()