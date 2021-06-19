import abc

import torchvision.transforms as T
import albumentations as A
from torch.utils.data import DataLoader, Dataset


class AugmentationFactoryBase(abc.ABC):
    def build_transforms(self, train):
        return self.build_train() if train else self.build_test()

    @abc.abstractmethod
    def build_train(self):
        pass

    @abc.abstractmethod
    def build_test(self):
        pass


class MNISTTransforms(AugmentationFactoryBase):

    def build_train(self):
        return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

    def build_test(self):
        return T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])


class CIFAR10Transforms(AugmentationFactoryBase):

    def build_train(self):
        return T.Compose([T.ToTensor(), A.Resize(200, 300),
            A.CenterCrop(100, 100),
            A.RandomCrop(80, 80),
            A.HorizontalFlip(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=45, p=.75),
            A.CoarseDropout(max_holes=1, max_height=16, max_width=1, min_holes = 1, min_height = 16, min_width = 16, fill_value = (0.49159187 ,0.48234594, 0.44671956), mask_fill_value = None),

            A.VerticalFlip(p=0.5),
            A.Normalize((0.49159187 ,0.48234594, 0.44671956), (0.23834434, 0.23486559, 0.25264624))])

    def build_test(self):
        return T.Compose([T.ToTensor(), A.Normalize((0.49159187 ,0.48234594, 0.44671956), (0.23834434, 0.23486559, 0.25264624))])
