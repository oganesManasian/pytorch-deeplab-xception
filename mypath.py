class Path(object):
    @staticmethod
    def db_root_dir(dataset):
        if dataset == 'cityscapes':
            # return 'data/cityscapes'
            return '../../../cvlabdata2/forOganes/cityscapes'
            # return '/path/to/datasets/cityscapes/'     # folder that contains leftImg8bit/
        elif dataset == 'cityscapes_local':
            return 'data/cityscapes'
        elif dataset == 'synthia':
            return 'data/synthia'
        # elif dataset == 'sbd':
        #     return '/path/to/datasets/benchmark_RELEASE/'  # folder that contains dataset/.
        # elif dataset == 'pascal':
        #     return '/path/to/datasets/VOCdevkit/VOC2012/'  # folder that contains VOCdevkit/.
        # elif dataset == 'coco':
        #     return '/path/to/datasets/coco/'
        else:
            print('Dataset {} not available.'.format(dataset))
            raise NotImplementedError
