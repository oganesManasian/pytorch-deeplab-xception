import argparse
import os
import numpy as np
from tqdm import tqdm
from PIL import Image

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
from dataloaders.utils import decode_segmap


def print_statistics(evaluator, label):
    print(f"Evaluating on {label} images")
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()

    print(f"Acc:{Acc:0.4f}, Acc_class:{Acc_class:0.4f}, mIoU:{mIoU:0.4f}, fwIoU: {FWIoU:0.4f}")

    acc_per_class = np.diag(evaluator.confusion_matrix) / evaluator.confusion_matrix.sum(axis=1)
    class_names = ['unlabelled', 'road', 'sidewalk', 'building', 'wall', 'fence',
                   'pole', 'traffic_light', 'traffic_sign', 'vegetation', 'terrain',
                   'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train',
                   'motorcycle', 'bicycle']
    class_performances = list(zip(class_names[1:], acc_per_class))
    for (class_name, perf) in class_performances:
        print(f"{class_name}: {perf:0.3f}")


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Predicting")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='cityscapes_local',
                        choices=['pascal', 'coco', 'cityscapes', 'synthia', 'cityscapes_local'],
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
    # parser.add_argument('--loss-type', type=str, default='ce',
    #                     choices=['ce', 'focal'],
    #                     help='loss func type (default: ce)')
    # training hyper params
    # parser.add_argument('--epochs', type=int, default=None, metavar='N',
    #                     help='number of epochs to train (default: auto)')
    # parser.add_argument('--start_epoch', type=int, default=0,
    #                     metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                predicting (default: auto)')
    # parser.add_argument('--use-balanced-weights', action='store_true', default=False,
    #                     help='whether to use balanced weights (default: False)')
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
    parser.add_argument('--val-data', type=str, default='val',
                        help='put the name of folder containing images to validate on')
    parser.add_argument('--test-data', type=str, default=None, required=True,
                        help='put the name of folder containing images to test on')
    parser.add_argument('--n_predictions', type=int, default=30,
                        help='Number of predictions to do')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

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

    # Predict
    model.eval()
    evaluator_real = Evaluator(nclass)
    evaluator_synthetic = Evaluator(nclass)

    tbar = tqdm(test_loader, desc='\r')
    folder_to_save = "predictions"
    if not os.path.isdir(folder_to_save):
        os.mkdir(folder_to_save)

    segmented = dict()
    real_n = 0
    synthetic_n = 0
    for i, sample in enumerate(tbar):
        image, target, path, is_synthetic = sample['image'], sample['label'], sample['path'][0], sample['is_synthetic'][0]

        label = "synthetic" if is_synthetic else "real"
        if label == 'real':
            real_n += 1
            if real_n > args.n_predictions:
                continue
        else:
            synthetic_n += 1
            if synthetic_n > args.n_predictions:
                continue

        if args.cuda:
            # image, target = image.cuda(), target.cuda()
            image, target = image.to(args.gpu_ids), target.to(args.gpu_ids)  # My change to make it work on cuda:3

        with torch.no_grad():
            output = model(image)

        # img = np.asarray(image)
        image = image.cpu().numpy().squeeze()
        image = np.transpose(image, [1, 2, 0])
        # image = np.swapaxes(image, 0, 1)
        # print(image.shape)
        # print(image.min(), image.max())

        pred = output.data.cpu().numpy().squeeze()
        pred = np.argmax(pred, axis=0)
        pred_rgb = decode_segmap(pred, args.dataset)
        # pred_rgb = (pred_rgb * 255).astype(np.uint8)

        filename = f"{path.split('.')[0].split('/')[-1]} {label}"
        segmented[filename] = pred_rgb

        # Calculate statistics
        target = target.cpu().numpy()
        pred = output.data.cpu().numpy()
        pred = np.argmax(pred, axis=1)
        if label == "real":
            evaluator_real.add_batch(target, pred)
        else:
            evaluator_synthetic.add_batch(target, pred)
    # print(pred_rgb.shape)
    # print(pred_rgb.min(), pred_rgb.max())

    # target = target.cpu().numpy().squeeze()

    # Combine real and segmented image
    # combined_image = np.zeros(shape=(image.shape[0], image.shape[1] * 2, image.shape[2]))
    # combined_image[:, :image.shape[1], :] = image
    # combined_image[:, image.shape[1]:, :] = pred_rgb
    # combined_image = Image.fromarray((combined_image * 255).astype(np.uint8))
    # combined_image.save(os.path.join(folder_to_save, filename))

    # if i >= args.n_predictions:
    #     break

    # print(segmented.keys())
    unique_scenes = np.unique([label.split(' ')[0] for label in segmented.keys()])

    for scene in unique_scenes:
        real = segmented[f"{scene} real"]
        synthetic = segmented[f"{scene} synthetic"]
        combined_image = np.zeros(shape=(real.shape[0], real.shape[1] * 2, real.shape[2]))
        combined_image[:, :real.shape[1], :] = real
        combined_image[:, real.shape[1]:, :] = synthetic
        combined_image = Image.fromarray((combined_image * 255).astype(np.uint8))
        combined_image.save(os.path.join(folder_to_save, f"{scene}.png"))

    print_statistics(evaluator_real, "real")
    print_statistics(evaluator_synthetic, "synthetic")


if __name__ == "__main__":
    main()
