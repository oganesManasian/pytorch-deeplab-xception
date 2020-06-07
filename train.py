# Experiment 8 - trained on Cityscapes day time (train )
# Experiment 9 - trained on Cityscapes day time + Cityscapes day time translated to night (balanced dataset) (train_combined2 2092 )
# Experiment 10 - trained on Cityscapes day time translated to night (train_bight)
# Experiment 11 - trained on Cityscapes day time + Cityscapes day time translated to night (GAN approach on balanced dataset) (tran_gan)
#
#

import argparse
import os
import numpy as np
from tqdm import tqdm

from mypath import Path
from dataloaders import make_data_loader
from modeling.sync_batchnorm.replicate import patch_replication_callback
from modeling.deeplab import *
from modeling.discriminator import DiscriminatorTorch
from utils.loss import SegmentationLosses, GANLosses
from utils.calculate_weights import calculate_weigths_labels
from utils.lr_scheduler import LR_Scheduler
from utils.saver import Saver
from utils.summaries import TensorboardSummary
from utils.metrics import Evaluator


class Trainer(object):

    def __init__(self, args):
        self.args = args

        # Define Saver
        self.saver = Saver(args)
        self.saver.save_experiment_config()
        # Define Tensorboard Summary
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Define Dataloader
        kwargs = {'num_workers': args.workers, 'pin_memory': True}
        self.train_loader, self.val_loader, self.test_loader, self.nclass = make_data_loader(args, **kwargs)

        # Define segmentation network
        model = DeepLab(num_classes=self.nclass,
                        backbone=args.backbone,
                        output_stride=args.out_stride,
                        sync_bn=args.sync_bn,
                        freeze_bn=args.freeze_bn)

        train_params = [{'params': model.get_1x_lr_params(), 'lr': args.lr},
                        {'params': model.get_10x_lr_params(), 'lr': args.lr * 10}]

        # Define Optimizer
        optimizer = torch.optim.SGD(train_params, momentum=args.momentum,
                                    weight_decay=args.weight_decay, nesterov=args.nesterov)

        # Define Criterion
        # whether to use class balanced weights
        if args.use_balanced_weights:
            classes_weights_path = os.path.join(Path.db_root_dir(args.dataset), args.dataset + '_classes_weights.npy')
            if os.path.isfile(classes_weights_path):
                weight = np.load(classes_weights_path)
            else:
                weight = calculate_weigths_labels(args.dataset, self.train_loader, self.nclass)
            weight = torch.from_numpy(weight.astype(np.float32))
        else:
            weight = None

        self.criterion = SegmentationLosses(weight=weight, cuda=args.cuda).build_loss(mode=args.loss_type)
        self.model, self.optimizer = model, optimizer

        # Using cuda
        self.gpu_ids = args.gpu_ids[0]
        if args.cuda:
            self.model = torch.nn.DataParallel(self.model, device_ids=self.args.gpu_ids)
            patch_replication_callback(self.model)
            # self.model = self.model.to(self.gpu_ids)  # My change to make it work on cuda:3
            self.model = self.model.to(self.gpu_ids)

        if args.use_gan:
            # Define discriminator (2 classes: real or fake)
            discriminator = DiscriminatorTorch(num_classes=2, use_pretrained=False)
            optimizer_discriminator = torch.optim.SGD(discriminator.parameters(),
                                                      lr=args.lr_discriminator,
                                                      momentum=args.momentum_discriminator,
                                                      weight_decay=args.weight_decay_discriminator)
            self.discriminator, self.optimizer_discriminator = discriminator, optimizer_discriminator
            self.criterion_gan = GANLosses(cuda=args.cuda)
            self.discriminator = self.discriminator.to(self.gpu_ids)

        # Define Evaluator
        self.evaluator = Evaluator(self.nclass)
        # Define lr scheduler
        self.scheduler = LR_Scheduler(args.lr_scheduler, args.lr,
                                      args.epochs, len(self.train_loader))

        # Resuming checkpoint
        self.best_pred = 0.0
        if args.resume is not None:
            if not os.path.isfile(args.resume):
                raise RuntimeError("=> no checkpoint found at '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.cuda:
                self.model.module.load_state_dict(checkpoint['state_dict'])
            else:
                self.model.load_state_dict(checkpoint['state_dict'])
            if not args.ft:
                self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.best_pred = checkpoint['best_pred']
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))

        # Clear start epoch if fine-tuning
        if args.ft:
            args.start_epoch = 0

    def training_segmentation(self, epoch):
        train_loss_total = 0.0
        generator_loss_total = 0.0
        self.model.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image, target, is_synthetic = sample['image'], sample['label'], sample['is_synthetic']
            if self.args.cuda:
                # My change to make it work on cuda:3
                image, target, is_synthetic = \
                    image.to(self.gpu_ids), target.to(self.gpu_ids), is_synthetic.to(self.gpu_ids)

            self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer.zero_grad()
            output = self.model(image)
            loss = self.criterion(output, target)

            if self.args.use_gan:
                output_on_fake = output[is_synthetic]  # Extract only segmentations done on synthetic images
                # with torch.no_grad():  # TODO check that freezes needed discriminator
                predicted_on_fake = self.discriminator(output_on_fake)
                loss_generator = self.criterion_gan.generator_loss(predicted_on_fake)
                generator_loss_total += loss_generator.item()
                loss += loss_generator

            loss.backward()
            self.optimizer.step()
            train_loss_total += loss.item()
            tbar.set_description('Segmentation + Generation loss: %.3f' % (train_loss_total / (i + 1)))
            # tbar.set_description('Segmentation + Generation loss: %.3f' % train_loss_total)
            self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)

            # Show 10 * 3 inference results each epoch
            if i % (num_img_tr // 10) == 0:
                global_step = i + num_img_tr * epoch
                self.summary.visualize_image(self.writer, self.args.dataset, image, target, output, global_step)

        self.writer.add_scalar('train/total_loss_epoch', train_loss_total, epoch)
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print(f"Segmentation + Generation loss: {train_loss_total:.3f}, Generator loss: {generator_loss_total:.3f}")

        if self.args.no_val:
            # save checkpoint every epoch
            is_best = False
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)

    def training_discriminant(self, epoch):
        discriminator_loss_total = 0.0
        self.discriminator.train()
        tbar = tqdm(self.train_loader)
        num_img_tr = len(self.train_loader)

        for i, sample in enumerate(tbar):
            image, is_synthetic = sample['image'], sample['is_synthetic']
            if self.args.cuda:
                # image, target = image.to(self.gpu_ids), target.to(self.gpu_ids)  # My change to make it work on cuda:3
                image, is_synthetic = image.to(self.gpu_ids), is_synthetic.to(self.gpu_ids)

            # # self.scheduler(self.optimizer, i, epoch, self.best_pred)
            self.optimizer_discriminator.zero_grad()
            # with torch.no_grad():  # TODO check that freezes needed generator
            output = self.model(image)
            output_on_real = output[~is_synthetic]
            output_on_fake = output[is_synthetic]
            predicted_on_real = self.discriminator(output_on_real)
            predicted_on_fake = self.discriminator(output_on_fake)

            loss = self.criterion_gan.discriminator_loss(predicted_on_real, predicted_on_fake)
            loss.backward()
            self.optimizer_discriminator.step()
            discriminator_loss_total += loss.item()
            tbar.set_description('Discriminator loss: %.3f' % (discriminator_loss_total / (i + 1)))
            # tbar.set_description('Discriminator loss: %.3f' % discriminator_loss_total)
            # self.writer.add_scalar('train/total_loss_iter', loss.item(), i + num_img_tr * epoch)
        print(f"Discriminator loss: {discriminator_loss_total:.3f}")

    def validation(self, epoch):
        self.model.eval()
        self.evaluator.reset()
        tbar = tqdm(self.val_loader, desc='\r')
        val_loss = 0.0
        for i, sample in enumerate(tbar):
            image, target = sample['image'], sample['label']
            if self.args.cuda:
                # image, target = image.cuda(), target.cuda()
                image, target = image.to(self.gpu_ids), target.to(self.gpu_ids)  # My change to make it work on cuda:3
            with torch.no_grad():
                output = self.model(image)
            loss = self.criterion(output, target)
            val_loss += loss.item()
            tbar.set_description('Validation loss: %.3f' % (val_loss / (i + 1)))
            # tbar.set_description('Validation loss: %.3f' % val_loss)
            pred = output.data.cpu().numpy()
            target = target.cpu().numpy()
            pred = np.argmax(pred, axis=1)
            # Add batch sample into evaluator
            self.evaluator.add_batch(target, pred)

        # Fast test during the training
        Acc = self.evaluator.Pixel_Accuracy()
        Acc_class = self.evaluator.Pixel_Accuracy_Class()
        mIoU = self.evaluator.Mean_Intersection_over_Union()
        FWIoU = self.evaluator.Frequency_Weighted_Intersection_over_Union()
        self.writer.add_scalar('val/total_loss_epoch', val_loss, epoch)
        self.writer.add_scalar('val/mIoU', mIoU, epoch)
        self.writer.add_scalar('val/Acc', Acc, epoch)
        self.writer.add_scalar('val/Acc_class', Acc_class, epoch)
        self.writer.add_scalar('val/fwIoU', FWIoU, epoch)
        # print('Validation:')
        print('[Epoch: %d, numImages: %5d]' % (epoch, i * self.args.batch_size + image.data.shape[0]))
        print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        print('Validation Loss (segmentation): %.3f' % val_loss)

        new_pred = mIoU
        if new_pred > self.best_pred:
            is_best = True
            self.best_pred = new_pred
            self.saver.save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': self.model.module.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'best_pred': self.best_pred,
            }, is_best)


def main():
    parser = argparse.ArgumentParser(description="PyTorch DeeplabV3Plus Training")
    parser.add_argument('--backbone', type=str, default='resnet',
                        choices=['resnet', 'xception', 'drn', 'mobilenet'],
                        help='backbone name (default: resnet)')
    parser.add_argument('--out-stride', type=int, default=16,
                        help='network output stride (default: 8)')
    parser.add_argument('--dataset', type=str, default='cityscapes',
                        choices=['pascal', 'coco', 'cityscapes', 'synthia'],
                        help='dataset name (default: cityscapes)')
    parser.add_argument('--use-sbd', action='store_true', default=False,
                        help='whether to use SBD dataset (default: False)')
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
    parser.add_argument('--epochs', type=int, default=None, metavar='N',
                        help='number of epochs to train (default: auto)')
    parser.add_argument('--start_epoch', type=int, default=0,
                        metavar='N', help='start epochs (default:0)')
    parser.add_argument('--batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                training (default: auto)')
    parser.add_argument('--test_batch-size', type=int, default=None,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    parser.add_argument('--use-balanced-weights', action='store_true', default=False,
                        help='whether to use balanced weights (default: False)')
    # optimizer params
    parser.add_argument('--lr', type=float, default=None, metavar='LR',
                        help='learning rate (default: auto)')
    parser.add_argument('--lr-scheduler', type=str, default='poly',
                        choices=['poly', 'step', 'cos'],
                        help='lr scheduler mode: (default: poly)')
    parser.add_argument('--momentum', type=float, default=0.9,
                        metavar='M', help='momentum (default: 0.9)')
    parser.add_argument('--weight-decay', type=float, default=5e-4,
                        metavar='M', help='w-decay (default: 5e-4)')
    parser.add_argument('--nesterov', action='store_true', default=False,
                        help='whether use nesterov (default: False)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, required=True,
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    parser.add_argument('--train-data', type=str, default=None, required=True,
                        help='put the name of folder containing images to train on')
    parser.add_argument('--val-data', type=str, default='val',
                        help='put the name of folder containing images to validate on')
    parser.add_argument('--test-data', type=str, default='val',
                        help='put the name of folder containing images to test on')
    # Gan params
    parser.add_argument('--use-gan', type=bool, default=False,
                        help='Whether to use discriminator for training')
    parser.add_argument('--lr-discriminator', type=float, default=0.01,
                        help='learning rate for discriminator (default: auto)')
    parser.add_argument('--momentum-discriminator', type=float, default=0.9,
                        help='momentum for discriminator (default: 0.9)')
    parser.add_argument('--weight-decay-discriminator', type=float, default=5e-4,
                        help='w-decay for discriminator (default: 5e-4)')

    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    if args.cuda:
        try:
            args.gpu_ids = [int(s) for s in args.gpu_ids.split(',')]
        except ValueError:
            raise ValueError('Argument --gpu_ids must be a comma-separated list of integers only')

    if args.sync_bn is None:
        if args.cuda and len(args.gpu_ids) > 1:
            args.sync_bn = True
        else:
            args.sync_bn = False

    # default settings for epochs, batch_size and lr
    if args.epochs is None:
        epoches = {
            'coco': 30,
            'cityscapes': 200,
            'pascal': 50,
            'synthia': 100,
        }
        args.epochs = epoches[args.dataset.lower()]

    if args.batch_size is None:
        args.batch_size = 4 * len(args.gpu_ids)

    if args.test_batch_size is None:
        args.test_batch_size = args.batch_size

    if args.lr is None:
        lrs = {
            'coco': 0.1,
            'cityscapes': 0.01,
            'pascal': 0.007,
            'synthia': 0.01
        }
        args.lr = lrs[args.dataset.lower()] / (4 * len(args.gpu_ids)) * args.batch_size

    if args.checkname is None:
        args.checkname = 'deeplab-' + str(args.backbone)
    print(args)
    torch.manual_seed(args.seed)
    trainer = Trainer(args)
    print('Starting Epoch:', trainer.args.start_epoch)
    print('Total Epoches:', trainer.args.epochs)
    for epoch in range(trainer.args.start_epoch, trainer.args.epochs):
        trainer.training_segmentation(epoch)
        # if args.use_gan and epoch % 20 == 0:
        if args.use_gan and epoch % 2 == 0:
            # Train discriminator once in 20 epoch for 5 epoch
            # for _ in range(5):
            for _ in range(1):
                trainer.training_discriminant(epoch)
        if not trainer.args.no_val and epoch % args.eval_interval == (args.eval_interval - 1):
            trainer.validation(epoch)

    trainer.writer.close()


if __name__ == "__main__":
    main()
