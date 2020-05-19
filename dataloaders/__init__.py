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
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class
    # elif args.dataset == 'synthia':
    #     train_set = synthia.SynthiaSegmentation(args, split='train')
    #     val_set = synthia.SynthiaSegmentation(args, split='val')
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None
    #     return train_loader, val_loader, test_loader, num_class


    #
    # elif args.dataset == 'pascal':
    #     train_set = pascal.VOCSegmentation(args, split='train')
    #     val_set = pascal.VOCSegmentation(args, split='val')
    #     if args.use_sbd:
    #         sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
    #         train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])
    #
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None
    #
    #     return train_loader, val_loader, test_loader, num_class
    #
    # elif args.dataset == 'coco':
    #     train_set = coco.COCOSegmentation(args, split='train')
    #     val_set = coco.COCOSegmentation(args, split='val')
    #     num_class = train_set.NUM_CLASSES
    #     train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
    #     val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
    #     test_loader = None
    #     return train_loader, val_loader, test_loader, num_class

    else:
        raise NotImplementedError
