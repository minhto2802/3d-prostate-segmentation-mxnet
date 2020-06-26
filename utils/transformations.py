import os

try:
    SEED = int(os.getenv('SEED'))
except:
    SEED = 0
import imgaug.augmenters as iaa
import numpy as np
from imgaug.random import seed

seed(SEED)
from imgaug.augmentables.segmaps import SegmentationMapsOnImage


def just_crop(input_size=320):
    seq = iaa.CropToFixedSize(position='center', width=input_size, height=input_size).to_deterministic()
    return seq


STATE = None


class Augmenter:
    """Define augmentation sequences"""

    def __init__(self, input_size=320):
        """Input shape always stay the same after the augmentation, while value be change for a same Augmenter object"""
        self.just_crop = iaa.CropToFixedSize(position='center', width=input_size, height=input_size)
        self.seq_shape = self.get_seq_shape(input_size).to_deterministic()  # iaa.Noop()
        self.seq_val = self.get_seq_val()  # iaa.Noop() self.get_seq_val()
        self.seq_val1 = self.get_seq_val()
        self.seq_val2 = self.get_seq_val()
        self.seq_noop = iaa.Sequential([iaa.Noop(), iaa.Noop()])

    def get_seq_combined(self, no_shape_augment=False, no_val_augment=False):
        """Same shape & same value augmentations every time"""
        seq = iaa.Sequential([
            self.seq_noop if no_shape_augment else self.seq_shape,
            self.seq_noop if no_val_augment else self.seq_val,
        ]).to_deterministic()
        return seq

    @staticmethod
    def get_seq_shape(input_size):
        sometimes = lambda aug: iaa.Sometimes(0.5, aug, random_state=STATE, )
        seq_shape = iaa.Sequential([
            # sometimes(iaa.Crop(percent=(0, .1))),  # crop images from each side by 0 to 16px (randomly chosen)
            sometimes(iaa.Fliplr(0.5, random_state=STATE, )),  # horizontally flip 50% of the images

            # Apply affine transformations to each image.
            # Scale/zoom them, translate/move them, rotate them and shear them.
            iaa.Affine(
                scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
                translate_percent={"x": (-0.02, 0.02), "y": (-0.02, 0.02)},
                rotate=(-25, 25),
                shear=(-8, 8),
                random_state=STATE,
            ),
            iaa.PerspectiveTransform(scale=(0.01, 0.10), random_state=STATE, ),
        ], random_order=True)
        return seq_shape

    @staticmethod
    def get_seq_val():
        sometimes = lambda aug: iaa.Sometimes(0.5, aug, random_state=STATE, )
        seq_val = iaa.Sequential([
            iaa.OneOf([
                sometimes(iaa.GaussianBlur(sigma=(0.1, 1), random_state=STATE, )),
                sometimes(iaa.AverageBlur(k=(3, 7))),
                sometimes(iaa.MotionBlur(k=(3, 7))),
                sometimes(iaa.AveragePooling((2, 8))),
                sometimes(iaa.MaxPooling((2, 8))),
                sometimes(iaa.MedianPooling((2, 8))),
                sometimes(iaa.MinPooling((2, 8))),
            ]),
            iaa.Multiply((0.8, 1.2)),
        ], random_order=True)
        return seq_val


def transform_sup(arrays, is_val=False, input_size=256, not_augment_values=False, num_slices=16):
    """"""
    # Specifically for PROMISE2012 inputs
    arrays = arrays[:, arrays[0].sum(axis=(1, 2)) > 0]
    arrays = list(arrays)
    if not is_val:
        arrays[0] = arrays[0].astype('float32')
        arrays[1] = arrays[1].astype('uint8')
        # Randomly select sequential slices from all slices
        num_slices = min(num_slices, arrays[0].shape[0])
        slice_idx_start = np.random.randint(0, arrays[0].shape[0] - num_slices + 1)
        arrays = [arr[slice_idx_start: slice_idx_start + num_slices] for arr in arrays]

        # Create augmentation sequences (shape + value)
        augmenter = Augmenter(input_size=input_size)  # always create the Augmenter object first
        seq_shape = augmenter.seq_shape
        seq_val = augmenter.seq_val

        tmp = [seq_shape(image=arr, segmentation_maps=SegmentationMapsOnImage(sm, shape=arr.shape)) for (arr, sm) in
               zip(arrays[0], arrays[1])]
        image_aug = np.asarray([seq_val(image=augmenter.just_crop(image=tmp1[0])) for tmp1 in tmp])
        segmap_aug = np.asarray([augmenter.just_crop(image=tmp1[1].get_arr()) for tmp1 in tmp])

        # Normalization
        _mean = image_aug.mean(axis=(1, 2)).reshape((-1,) + (1,) * 2)
        _std = image_aug.std(axis=(1, 2)).reshape((-1,) + (1,) * 2)
        image_aug = (image_aug - _mean) / _std

        return (
            image_aug[np.newaxis],  # add channel axis
            segmap_aug,
        )
    else:
        augmenter = Augmenter(input_size=input_size)
        arrays[0] = np.asarray([augmenter.just_crop(image=tmp) for tmp in arrays[0]])
        _mean = arrays[0].mean(axis=(1, 2)).reshape((-1,) + (1,) * 2)
        _std = arrays[0].std(axis=(1, 2)).reshape((-1,) + (1,) * 2)
        arrays[0] = (arrays[0] - _mean) / _std  # Normalization
        arrays[0] = arrays[0][np.newaxis]  # add channel axis
        if len(arrays) == 2:
            arrays[1] = np.asarray([augmenter.just_crop(image=tmp) for tmp in arrays[1]]).astype('float32')
        return tuple(arrays)


def joint_transform(arrays, is_val=False, input_size=256, not_augment_values=False):
    """"""
    return transform_sup(arrays, is_val=is_val, input_size=input_size, not_augment_values=not_augment_values)
