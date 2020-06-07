import torch
import torch.nn as nn


class SegmentationLosses(object):
    def __init__(self, weight=None, size_average=True, batch_average=True, ignore_index=255, cuda=False):
        self.ignore_index = ignore_index
        self.weight = weight
        self.size_average = size_average
        self.batch_average = batch_average
        self.cuda = cuda

    def build_loss(self, mode='ce'):
        """Choices: ['ce' or 'focal']"""
        if mode == 'ce':
            return self.CrossEntropyLoss
        elif mode == 'focal':
            return self.FocalLoss
        else:
            raise NotImplementedError

    def CrossEntropyLoss(self, logit, target):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        loss = criterion(logit, target.long())

        if self.batch_average:
            loss /= n

        return loss

    def FocalLoss(self, logit, target, gamma=2, alpha=0.5):
        n, c, h, w = logit.size()
        criterion = nn.CrossEntropyLoss(weight=self.weight, ignore_index=self.ignore_index,
                                        size_average=self.size_average)
        if self.cuda:
            criterion = criterion.cuda()

        logpt = -criterion(logit, target.long())
        pt = torch.exp(logpt)
        if alpha is not None:
            logpt *= alpha
        loss = -((1 - pt) ** gamma) * logpt

        if self.batch_average:
            loss /= n

        return loss


class GANLosses(object):
    """
    Implementation of non saturating GAN loss
    """

    def __init__(self, cuda):
        self.criterion = torch.nn.CrossEntropyLoss()
        if cuda:
            self.criterion = self.criterion.cuda()

    def generator_loss(self, predicted_on_fake):
        """
        Maximise probability of generated images to be predicted as real
        :param predicted_on_fake: discriminator predictions for generated images
        :return: loss
        """
        target = torch.full((predicted_on_fake.size(0),),
                            fill_value=1,
                            device=predicted_on_fake.device,
                            dtype=torch.long)
        # print(f"Generator loss {self.criterion(predicted_on_fake, target)}, "
        #       f"target/predicted fake: {target}/{predicted_on_fake},")
        return self.criterion(predicted_on_fake, target)

    def discriminator_loss(self, predicted_on_real, predicted_on_fake):
        """
        Maximise probability of real images to be predicted as real and generated images to be predicted as fake
        :param predicted_on_real: discriminator predictions for real images
        :param predicted_on_fake: discriminator predictions for generated images
        :return: loss
        """
        target_for_real = torch.full((predicted_on_real.size(0),),
                                     fill_value=1,
                                     device=predicted_on_real.device,
                                     dtype=torch.int64)
        target_for_fake = torch.full_like(target_for_real, fill_value=0)

        # print(f"Discriminator loss {(self.criterion(predicted_on_real, target_for_real) + self.criterion(predicted_on_fake, target_for_fake)) / 2}, "
        #       f"target/predicted real: {target_for_real}/{predicted_on_real},"
        #       f"target/predicted fake: {target_for_fake}/{predicted_on_fake},")
        return (self.criterion(predicted_on_real, target_for_real)
                + self.criterion(predicted_on_fake, target_for_fake)) / 2


if __name__ == "__main__":
    loss = SegmentationLosses(cuda=True)
    a = torch.rand(1, 3, 7, 7).cuda()
    b = torch.rand(1, 7, 7).cuda()
    print(loss.CrossEntropyLoss(a, b).item())
    print(loss.FocalLoss(a, b, gamma=0, alpha=None).item())
    print(loss.FocalLoss(a, b, gamma=2, alpha=0.5).item())
