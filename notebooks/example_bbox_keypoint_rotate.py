from typing import List

import albumentations as A
import cv2
import matplotlib.pyplot as plt
import numpy as np


def visualize(image: np.ndarray, keypoints: List[List[float]], bboxes: List[List[float]]) -> np.ndarray:
    overlay = image.copy()
    for kp in keypoints:
        cv2.circle(overlay, (int(kp[0]), int(kp[1])), 20, (0, 200, 200), thickness=2, lineType=cv2.LINE_AA)

    for box in bboxes:
        cv2.rectangle(overlay, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (200, 0, 0), thickness=2)

    return overlay


def main() -> None:
    image = cv2.imread("images/image_1.jpg")

    keypoints = cv2.goodFeaturesToTrack(
        cv2.cvtColor(image, cv2.COLOR_RGB2GRAY), maxCorners=100, qualityLevel=0.5, minDistance=5
    ).squeeze(1)

    bboxes = [(kp[0] - 10, kp[1] - 10, kp[0] + 10, kp[1] + 10) for kp in keypoints]

    disp_image = visualize(image, keypoints, bboxes)
    plt.figure(figsize=(10, 10))
    plt.imshow(cv2.cvtColor(disp_image, cv2.COLOR_RGB2BGR))
    plt.tight_layout()
    plt.show()

    aug = A.Compose(
        [A.Affine(scale=0.5, shear=10, translate_percent=0.2, rotate=10)],
        bbox_params=A.BboxParams(format="pascal_voc", label_fields=["bbox_labels"]),
        keypoint_params=A.KeypointParams(format="xy"),
        strict=True,
    )

    for _ in range(10):
        data = aug(image=image, keypoints=keypoints, bboxes=bboxes, bbox_labels=np.ones(len(bboxes)))

        aug_image = data["image"]
        aug_image = visualize(aug_image, data["keypoints"], data["bboxes"])

        plt.figure(figsize=(10, 10))
        plt.imshow(cv2.cvtColor(aug_image, cv2.COLOR_RGB2BGR))
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
