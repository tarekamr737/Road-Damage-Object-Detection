from __future__ import annotations

import numpy as np

from road_damage.evaluation.detection import box_iou_xyxy, matched_ious


def test_box_iou_xyxy_identical_boxes() -> None:
    box = np.asarray([0, 0, 10, 10], dtype=float)
    assert box_iou_xyxy(box, box) == 1.0


def test_box_iou_xyxy_partial_overlap() -> None:
    first = np.asarray([0, 0, 10, 10], dtype=float)
    second = np.asarray([5, 5, 15, 15], dtype=float)
    assert round(box_iou_xyxy(first, second), 4) == 0.1429


def test_matched_ious_requires_matching_class() -> None:
    pred_boxes = np.asarray([[0, 0, 10, 10]], dtype=float)
    pred_classes = np.asarray([1])
    pred_conf = np.asarray([0.99])
    gt_boxes = np.asarray([[0, 0, 10, 10]], dtype=float)
    gt_classes = np.asarray([2])

    ious, classes = matched_ious(
        pred_boxes,
        pred_classes,
        pred_conf,
        gt_boxes,
        gt_classes,
        threshold=0.5,
    )

    assert ious == []
    assert classes == []
