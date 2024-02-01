from __future__ import division, absolute_import, print_function
import argparse

def main(args):
    assert args.dataset in ['mnist', 'drebin', 'svhn', 'tiny', 'tiny_gray'], \
        "dataset parameter must be either 'mnist', 'cifar_cnn', 'cifar_densenet', 'svhn', or 'tiny'"
    print('Data set: %s' % args.dataset)
    
    if args.dataset == 'drebin':
        from baselineCNN.nn.nn_drebin import DREBINNN as model
        model_mnist = model(mode='train', filename='nn_{}.h5'.format(args.dataset), epochs=args.epochs, batch_size=args.batch_size)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-d', '--dataset',
        help="Dataset to use; either 'mnist', 'cifar', 'svhn', or 'tiny'",
        required=True, type=str
    )
    parser.add_argument(
        '-e', '--epochs',
        help="The number of epochs to train for.",
        required=False, type=int
    )
    parser.add_argument(
        '-b', '--batch_size',
        help="The batch size to use for training.",
        required=False, type=int
    )

    parser.set_defaults(epochs=50)
    parser.set_defaults(batch_size=16)
    args = parser.parse_args()
    main(args)
