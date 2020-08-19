# Notebooks

- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/example.ipynb) [example.ipynb](notebooks/example.ipynb). Defining a simple augmentation pipeline for image augmentation.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/example_16_bit_tiff.ipynb) [example_16_bit_tiff.ipynb](notebooks/example_16_bit_tiff.ipynb). Working with non-8-bit images.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/example_bboxes.ipynb) [example_bboxes.ipynb](notebooks/example_bboxes.ipynb). Using Albumentations to augment bounding boxes for object detection tasks.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/example_bboxes2.ipynb) [example_bboxes2.ipynb](notebooks/example_bboxes2.ipynb). How to use Albumentations for detection tasks if you need to keep all bounding boxes.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/example_kaggle_salt.ipynb) [example_kaggle_salt.ipynb](notebooks/example_kaggle_salt.ipynb). Using Albumentations for a semantic segmentation task.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/example_keypoints.ipynb) [example_keypoints.ipynb](notebooks/example_keypoints.ipynb). Using Albumentations to augment keypoints.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/example_multi_target.ipynb) [example_multi_target.ipynb](notebooks/example_multi_target.ipynb). Applying the same augmentation with the same parameters to multiple images, masks, bounding boxes, or keypoints.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/example_weather_transforms.ipynb) [example_weather_transforms.ipynb](notebooks/example_weather_transforms.ipynb). Weather augmentations in Albumentations.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/migrating_from_torchvision_to_albumentations.ipynb) [migrating_from_torchvision_to_albumentations.ipynb](notebooks/migrating_from_torchvision_to_albumentations.ipynb). Migrating from torchvision to Albumentations.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/replay.ipynb) [replay.ipynb](notebooks/replay.ipynb). Debugging an augmentation pipeline with ReplayCompose.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/serialization.ipynb) [serialization.ipynb](notebooks/serialization.ipynb). How to save and load parameters of an augmentation pipeline.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/showcase.ipynb) [showcase.ipynb](notebooks/showcase.ipynb). Showcase. Cool augmentation examples on diverse set of images from various real-world tasks.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/tensorflow-example.ipynb) [tensorflow-example.ipynb](notebooks/tensorflow-example.ipynb). Using Albumentations with Tensorflow.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/pytorch_classification.ipynb) [pytorch_classification.ipynb](notebooks/pytorch_classification.ipynb). PyTorch and Albumentations for image classification.
- [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/albumentations-team/albumentations_examples/blob/colab/pytorch_semantic_segmentation.ipynb) [pytorch_semantic_segmentation.ipynb](notebooks/pytorch_semantic_segmentation.ipynb). PyTorch and Albumentations for semantic segmentation.


# Usage examples

For detailed examples see [notebooks](https://github.com/albumentations-team/albumentations_examples/tree/master/notebooks).

```
from albumentations import (
    HorizontalFlip, IAAPerspective, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    IAAAdditiveGaussianNoise, GaussNoise, MotionBlur, MedianBlur, IAAPiecewiseAffine,
    IAASharpen, IAAEmboss, RandomBrightnessContrast, Flip, OneOf, Compose
)
import numpy as np

def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        OneOf([
            IAAAdditiveGaussianNoise(),
            GaussNoise(),
        ], p=0.2),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1),
            IAAPiecewiseAffine(p=0.3),
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
            IAASharpen(),
            IAAEmboss(),
            RandomBrightnessContrast(),
        ], p=0.3),
        HueSaturationValue(p=0.3),
    ], p=p)

image = np.ones((300, 300, 3), dtype=np.uint8)
mask = np.ones((300, 300), dtype=np.uint8)
whatever_data = "my name"
augmentation = strong_aug(p=0.9)
data = {"image": image, "mask": mask, "whatever_data": whatever_data, "additional": "hello"}
augmented = augmentation(**data)
image, mask, whatever_data, additional = augmented["image"], augmented["mask"], augmented["whatever_data"], augmented["additional"]
```

# Augmentations examples.

[MultiplicativeNoise]([MultiplicativeNoise](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.MultiplicativeNoise))
-------------------

1. Original image
2. `MultiplicativeNoise(multiplier=0.5, p=1)`
3. `MultiplicativeNoise(multiplier=1.5, p=1)`
4. `MultiplicativeNoise(multiplier=[0.5, 1.5], per_channel=True, p=1)`
5. `MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, p=1)`
6. `MultiplicativeNoise(multiplier=[0.5, 1.5], elementwise=True, per_channel=True, p=1)`

![MultiplicativeNoise image](images/augs_examples/MultiplicativeNoise.jpg)


[ToSepia]([ToSepia](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ToSepia))
-------

1. Original image
2. `ToSepia(p=1)`

![ToSepia image](images/augs_examples/ToSepia.jpg)


[JpegCompression]([JpegCompression](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.JpegCompression))
---------------

1. Original image
2. `JpegCompression(quality_lower=99, quality_upper=100, p=1)`
3. `JpegCompression(quality_lower=59, quality_upper=60, p=1)`
4. `JpegCompression(quality_lower=39, quality_upper=40, p=1)`
5. `JpegCompression(quality_lower=19, quality_upper=20, p=1)`
6. `JpegCompression(quality_lower=0, quality_upper=1, p=1)`

![JpegCompression image](images/augs_examples/JpegCompression.jpg)


[ChannelDropout]([ChannelDropout](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ChannelDropout))
--------------

1. Original image
2. `ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1)`
3. `ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1)`
4. `ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1)`
5. `ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1)`
6. `ChannelDropout(channel_drop_range=(1, 1), fill_value=0, p=1)`
7. `ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=1)`
8. `ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=1)`
9. `ChannelDropout(channel_drop_range=(2, 2), fill_value=0, p=1)`

![ChannelDropout image](images/augs_examples/ChannelDropout.jpg)


[ChannelShuffle]([ChannelShuffle](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ChannelShuffle))
--------------

1. Original image
2. `ChannelShuffle(p=1)`
3. `ChannelShuffle(p=1)`
4. `ChannelShuffle(p=1)`

![ChannelShuffle image](images/augs_examples/ChannelShuffle.jpg)


[Cutout]([Cutout](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Cutout))
------

1. Original image
2. `Cutout(num_holes=8, max_h_size=8, max_w_size=8, fill_value=0, p=1)`
3. `Cutout(num_holes=10, max_h_size=20, max_w_size=20, fill_value=0, p=1)`
4. `Cutout(num_holes=30, max_h_size=30, max_w_size=30, fill_value=64, p=1)`
5. `Cutout(num_holes=50, max_h_size=40, max_w_size=40, fill_value=128, p=1)`
6. `Cutout(num_holes=100, max_h_size=50, max_w_size=50, fill_value=255, p=1)`

![Cutout image](images/augs_examples/Cutout.jpg)


[ToGray]([ToGray](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.ToGray))
------

1. Original image
2. `ToGray(p=1)`

![ToGray image](images/augs_examples/ToGray.jpg)


[InvertImg]([InvertImg](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.InvertImg))
---------

1. Original image
2. `InvertImg(p=1)`

![InvertImg image](images/augs_examples/InvertImg.jpg)


[VerticalFlip]([VerticalFlip](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.VerticalFlip))
------------

1. Original image
2. `VerticalFlip(p=1)`

![VerticalFlip image](images/augs_examples/VerticalFlip.jpg)


[HorizontalFlip]([HorizontalFlip](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.HorizontalFlip))
--------------

1. Original image
2. `HorizontalFlip(p=1)`

![HorizontalFlip image](images/augs_examples/HorizontalFlip.jpg)


[Flip]([Flip](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Flip))
----

1. Original image
2. `Flip(p=1)`
3. `Flip(p=1)`
4. `Flip(p=1)`

![Flip image](images/augs_examples/Flip.jpg)


[RandomGridShuffle]([RandomGridShuffle](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.RandomGridShuffle))
-----------------

1. Original image
2. `RandomGridShuffle(grid=(3, 3), p=1)`
3. `RandomGridShuffle(grid=(5, 5), p=1)`
4. `RandomGridShuffle(grid=(7, 7), p=1)`

![RandomGridShuffle image](images/augs_examples/RandomGridShuffle.jpg)


[Blur]([Blur](https://albumentations.readthedocs.io/en/latest/api/augmentations.html#albumentations.augmentations.transforms.Blur))
----

1. Original image
2. `Blur(blur_limit=(7, 7), p=1)`
3. `Blur(blur_limit=(15, 15), p=1)`
4. `Blur(blur_limit=(50, 50), p=1)`
5. `Blur(blur_limit=(100, 100), p=1)`
6. `Blur(blur_limit=(300, 300), p=1)`

![Blur image](images/augs_examples/Blur.jpg)
