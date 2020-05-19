import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from utils.loss import SegmentationLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Testing")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal', 'coco', 'cityscapes', 'synthia'],
                        help='dataset name (default: cityscapes)')
    # parser.add_argument('--use-sbd', action='store_true', default=False,
    #                     help='whether to use SBD dataset (default: False)')
    parser.add_argument('--workers', type=int, default=4,
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=513,
                        help='base image size')
    parser.add_argument('--crop-size', type=int, default=513,
                        help='crop image size')
    parser.add_argument('--sync-bn', type=bool, default=None,
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # training hyper params
    # parser.add_argument('--epochs', type=int, default=None, metavar='N',
    #                     help='number of epochs to train (default: auto)')
    # parser.add_argument('--start_epoch', type=int, default=0,
    #                     metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test_batch_size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    # parser.add_argument('--lr', type=float, default=None, metavar='LR',
    #                     help='learning rate (default: auto)')
    # parser.add_argument('--lr-scheduler', type=str, default='poly',
    #                     choices=['poly', 'step', 'cos'],
    #                     help='lr scheduler mode: (default: poly)')
    # parser.add_argument('--momentum', type=float, default=0.9,
    #                     metavar='M', help='momentum (default: 0.9)')
    # parser.add_argument('--weight-decay', type=float, default=5e-4,
    #                     metavar='M', help='w-decay (default: 5e-4)')
    # parser.add_argument('--nesterov', action='store_true', default=False,
    #                     help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, required=True,
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None, required=True,
                        help='put the path to resuming file if needed')
    # parser.add_argument('--checkname', type=str, default=None,
    #                     help='set the checkpoint name')
    # # finetuning pre-trained models
    # parser.add_argument('--ft', action='store_true', default=False,
    #                     help='finetuning on a different dataset')
    # # evaluation option
    # parser.add_argument('--eval-interval', type=int, default=1,
    #                     help='evaluation interval (default: 1)')
    # parser.add_argument('--no-val', action='store_true', default=False,
    #                     help='skip validation during training')
    # Define on what to test
    parser.add_argument('--train-data', type=str, default="train",
                        help='put the name of folder containing images to train on')
    parser.add_argument('--test-data', type=str, default=None, required=True,
                        help='put the name of folder containing images to test on')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    print(args)

    # Define Dataloader
    kwargs = {'num_workers': args.workers, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)

    # Built model
    model = DeepLab(num_classes=nclass,
                    backbone=args.backbone,
                    output_stride=args.out_stride,
                    sync_bn=args.sync_bn,
                    freeze_bn=args.freeze_bn)

    # Move model to gpu
    if args.cuda:
        model = torch.nn.DataParallel(model, device_ids=args.gpu_ids)
        patch_replication_callback(model)
        # model = model.cuda()
        args.gpu_ids = args.gpu_ids[0]
        model = model.to(args.gpu_ids)  # My change to make it work on cuda:3

    # Load checkpoint
    if args.resume is None:
        raise RuntimeError("No checkpoint provided")

    if not os.path.isfile(args.resume):
        raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))

    checkpoint = torch.load(args.resume)
    if args.cuda:
        model.module.load_state_dict(checkpoint['state_dict'])
    else:
        model.load_state_dict(checkpoint['state_dict'])

    print("=> loaded checkpoint '{}' (epoch {})"
          .format(args.resume, checkpoint['epoch']))

    # Define criterion
    if args.use_balanced_weights:
        classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
        if os.path.isfile(classes_weights_path):
            weight = np.load(classes_weights_path)
        else:
            weight = calculate_weigths_labels(args.dataset, train_loader, nclass)
        weight = torch.from_numpy(weight.astype(np.float32))
    else:
        weight = None
    criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)


    # Test
    model.eval()
    evaluator = Evaluator(nclass)
    # evaluator.reset()
    # tbar = tqdm(val_loader, desc='\r')
    tbar = tqdm(test_loader, desc='\r')
    test_loss = 0.0
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
        if args.cuda:
            # image, target = image.cuda(), target.cuda()
            image, target = image.to(args.gpu_ids), target.to(args.gpu_ids)  # My change to make it work on cuda:3
        with torch.no_grad():
            output = model(image)
        loss = criterion(output, target)
        test_loss += loss.item()
        tbar.set_description('Test loss: %.3f' % (test_loss / (i + 1)))
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        evaluator.add_batch(target, pred)

    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    print('Testing model:')
    # print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    print(f"Acc:{Acc:0.4f}, Acc_class:{Acc_class:0.4f}, mIoU:{mIoU:0.4f}, fwIoU: {FWIoU:0.4f}")
    print('Loss: %.3f' % test_loss)
    acc_per_class = np.diag(evaluator.confusion_matrix) / evaluator.confusion_matrix.sum(axis=1)
    class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                   'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                   'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                   'motorcycle', 'bicycle']
    class_performances = list(zip(class_names[1:], acc_per_class))
    for (class_name, perf) in class_performances:
        print(f"{class_name}: {perf:0.3f}")


if __name__ == "__main__":
    main()
