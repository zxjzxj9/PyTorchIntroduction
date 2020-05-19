import argparse

parser = argparse.ArgumentParser(description='PyTorch LeNet Training')
parser.add_argument('--lr', default=0.01, type=float, help='Learning rate')
parser.add_argument('--batch-size', '-b', default=256, type=int,
    help='Batchsize')
args = parser.parse_args()

print(args.lr, args.batch_size)