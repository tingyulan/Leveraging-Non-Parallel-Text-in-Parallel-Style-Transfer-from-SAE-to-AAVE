#! /usr/bin/env python
import os
import argparse
import datetime
import torch
# import torchtext.data as data
from torchtext.legacy import data
import torchtext.datasets as datasets
import model
import train
import mydatasets
import dill


parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=64, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data 
parser.add_argument('-shuffle', action='store_true', default=True, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=16, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3', help='comma-separated kernel size to use for convolution [default: 3,4,5]')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-field_dir', type=str, default=None, help='[default: None]')
parser.add_argument('-pred_path', type=str, default=None, help='[default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
args = parser.parse_args()

# load AAVE_SAE dataset
def aave_sae(text_field, label_field, root=".", step="train", **kargs):
    aave_sae_data = mydatasets.AAVE_SAE.splits(text_field, label_field, root=root, step=step)

    if step is "train":
        text_field.build_vocab(aave_sae_data)
        label_field.build_vocab(aave_sae_data)
        aave_sae_iter = data.Iterator(dataset=aave_sae_data, batch_size=args.batch_size, shuffle=True, **kargs)
    else:
        aave_sae_iter = data.BucketIterator(dataset=aave_sae_data, batch_size=args.batch_size, shuffle=False, **kargs)
    
    return aave_sae_iter


# load data
print("\nLoading data...")

if not args.test:
    data_root = "./../data/TextCNN_classifier"
    text_field = data.Field(lower=True)
    label_field = data.Field(sequential=False)

    train_iter = aave_sae(text_field, label_field, data_root, step="train")
    dev_iter = aave_sae(text_field, label_field, data_root, step="val")
else:
    with open(os.path.join(args.field_dir, "text_field.Field"), "rb") as f:
        text_field = dill.load(f)
    with open(os.path.join(args.field_dir, "label_field.Field"), "rb") as f:
        label_field = dill.load(f)

    test_iter = aave_sae(text_field, label_field, args.pred_path, step="test")


# update args and print
args.embed_num = len(text_field.vocab)
args.class_num = len(label_field.vocab) - 1
args.cuda = (not args.no_cuda) and torch.cuda.is_available(); del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

print("\nParameters:")
for attr, value in sorted(args.__dict__.items()):
    print("\t{}={}".format(attr.upper(), value))

# model
cnn = model.CNN_Text(args)
if args.snapshot is not None:
    print('\nLoading model from {}...'.format(args.snapshot))
    cnn.load_state_dict(torch.load(args.snapshot))

if args.cuda:
    torch.cuda.set_device(args.device)
    cnn = cnn.cuda()

# train or predict
if args.predict is not None:
    label = train.predict(args.predict, cnn, text_field, label_field, args.cuda)
    print('\n[Text]  {}\n[Label] {}\n'.format(args.predict, label))
elif args.test:
    try:
        train.eval(test_iter, cnn, args) 
    except Exception as e:
        print("\nSorry. The test dataset doesn't  exist.\n")
else:
    print()
    try:
        train.train(train_iter, dev_iter, cnn, args)
    except KeyboardInterrupt:
        print('\n' + '-' * 89)
        print('Exiting from training early')

    with open(os.path.join(args.save_dir, "text_field.Field"), "wb") as f:
        dill.dump(text_field, f)
    with open(os.path.join(args.save_dir, "label_field.Field"), "wb") as f:
        dill.dump(label_field, f)

