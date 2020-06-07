# from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
from dataloaders.datasets import cityscapes # , synthia
from torch.utils.data import DataLoader


def make_data_loader(args, **kwargs):
    if args.dataset == 'cityscapes' or args.dataset == 'cityscapes_local':
        train_set = cityscapes.CityscapesSegmentation(args, split=args.train_data)
        # val_set = cityscapes.CityscapesSegmentation(args, split='val')
        val_set = cityscapes.CityscapesSegmentation(args, split=args.val_data)
        # test_set = cityscapes.CityscapesSegmentation(args, split='test')
        test_set = cityscapes.CityscapesSegmentation(args, split=args.test_data)  # My change

        num_class = train_set.NUM_CLASSES
        if args.use_gan:
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        else:
            train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError
