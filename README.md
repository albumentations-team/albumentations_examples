# Notebooks

If you found errors in the notebooks, please create:

1. Pull request with the fix
or
2. Submit an issue to the [main repository of Albumentations](https://github.com/albumentations-team/albumentations/issues)

- [example.ipynb](notebooks/example.ipynb). Defining a simple augmentation pipeline for image augmentation.
- [example_16_bit_tiff.ipynb](notebooks/example_16_bit_tiff.ipynb). Working with non-8-bit images.
- [example_bboxes.ipynb](notebooks/example_bboxes.ipynb). Using Albumentations to augment bounding boxes for object detection tasks.
- [example_bboxes2.ipynb](notebooks/example_bboxes2.ipynb). How to use Albumentations for detection tasks if you need to keep all bounding boxes.
- [example_kaggle_salt.ipynb](notebooks/example_kaggle_salt.ipynb). Using Albumentations for a semantic segmentation task.
- [example_keypoints.ipynb](notebooks/example_keypoints.ipynb). Using Albumentations to augment keypoints.
- [example_multi_target.ipynb](notebooks/example_multi_target.ipynb). Applying the same augmentation with the same parameters to multiple images, masks, bounding boxes, or keypoints.
- [example_weather_transforms.ipynb](notebooks/example_weather_transforms.ipynb). Weather augmentations in Albumentations.
- [migrating_from_torchvision_to_albumentations.ipynb](notebooks/migrating_from_torchvision_to_albumentations.ipynb). Migrating from torchvision to Albumentations.
- [replay.ipynb](notebooks/replay.ipynb). Debugging an augmentation pipeline with ReplayCompose.
- [serialization.ipynb](notebooks/serialization.ipynb). How to save and load parameters of an augmentation pipeline.
- [showcase.ipynb](notebooks/showcase.ipynb). Showcase. Cool augmentation examples on diverse set of images from various real-world tasks.
- [tensorflow-example.ipynb](notebooks/tensorflow-example.ipynb). Using Albumentations with Tensorflow.
- [pytorch_classification.ipynb](notebooks/pytorch_classification.ipynb). PyTorch and Albumentations for image classification.
- [pytorch_semantic_segmentation.ipynb](notebooks/pytorch_semantic_segmentation.ipynb). PyTorch and Albumentations for semantic segmentation.
- [xy_transform.ipynb](notebooks/example_xymasking.ipynb). How to apply [XYMasking](https://albumentations.ai/docs/api_reference/augmentations/dropout/xy_masking/#xymasking-augmentation-augmentationsdropoutxy_masking).

## Usage examples

For detailed examples see [notebooks](https://github.com/albumentations-team/albumentations_examples/tree/master/notebooks).

```python
from albumentations import (
    HorizontalFlip, ShiftScaleRotate, CLAHE, RandomRotate90,
    Transpose, ShiftScaleRotate, Blur, OpticalDistortion, GridDistortion, HueSaturationValue,
    GaussNoise, MotionBlur, MedianBlur,
    RandomBrightnessContrast, Flip, OneOf, Compose
)
import numpy as np

def strong_aug(p=0.5):
    return Compose([
        RandomRotate90(),
        Flip(),
        Transpose(),
        GaussNoise(),
        OneOf([
            MotionBlur(p=0.2),
            MedianBlur(blur_limit=3, p=0.1),
            Blur(blur_limit=3, p=0.1),
        ], p=0.2),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.2, rotate_limit=45, p=0.2),
        OneOf([
            OpticalDistortion(p=0.3),
            GridDistortion(p=0.1)
        ], p=0.2),
        OneOf([
            CLAHE(clip_limit=2),
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

## Augmentations examples

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
