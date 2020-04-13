import sys
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='')

    #  experiment settings
    parser.add_argument('--root_dir', default='../ModelNet10/', type=str,
                            help='dataset directory')
    parser.add_argument('--batch_size', default=32, type=int,
                            help='training batch size')
    parser.add_argument('--lr', default=1e-3, type=float,
                            help='learning rate')
    parser.add_argument('--epochs', default=15, type=int,
                            help='number of training epochs')
    parser.add_argument('--save_model_path', default='./checkpoints/', type=str,
                            help='checkpoints dir')


    args = parser.parse_args()
    
    assert args.root_dir is not None
    
    print(' '.join(sys.argv))
    print(args)

    return args
