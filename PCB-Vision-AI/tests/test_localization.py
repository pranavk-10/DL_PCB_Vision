"""Test localization geometry and matching without requiring a trained model."""
import unittest
from src.dataset_loader import Annotation
from src.localization import Detection, box_iou, match_detections, non_max_suppression, window_positions

class LocalizationGeometryTests(unittest.TestCase):
    def test_iou_identical_disjoint_and_partial(self):
        self.assertEqual(box_iou((0,0,10,10),(0,0,10,10)),1)
        self.assertEqual(box_iou((0,0,10,10),(10,0,20,10)),0)
        self.assertAlmostEqual(box_iou((0,0,10,10),(5,0,15,10)),1/3)
    def test_iou_invalid_box(self):
        with self.assertRaises(ValueError):box_iou((0,0,0,3),(0,0,3,3))
    def test_nms_keeps_highest_score_within_class(self):
        first=Detection((0,0,10,10),0,0.9);second=Detection((1,1,11,11),0,0.8)
        self.assertEqual(non_max_suppression([second,first]),[first])
    def test_nms_preserves_distinct_classes(self):
        values=[Detection((0,0,10,10),0,0.9),Detection((0,0,10,10),1,0.8)]
        self.assertEqual(len(non_max_suppression(values)),2)
    def test_normal_and_nan_scores_rejected(self):
        for item in (Detection((0,0,10,10),6,0.9),Detection((0,0,10,10),0,float('nan'))):
            with self.subTest(item=item),self.assertRaises(ValueError):non_max_suppression([item])
    def test_window_edges_and_small_images(self):
        self.assertEqual(window_positions(640,128,64),list(range(0,513,64)))
        self.assertEqual(window_positions(650,128,64)[-1],522)
        self.assertEqual(window_positions(50,128,64),[0])
    def test_invalid_stride_rejected(self):
        with self.assertRaises(ValueError):window_positions(640,128,129)
    def test_one_to_one_matching_counts_duplicates_as_false_positive(self):
        truth=[Annotation((0,0,10,10),0,1)]
        predictions=[Detection((0,0,10,10),0,0.9),Detection((0,0,10,10),0,0.8)]
        result=match_detections(predictions,truth)
        self.assertEqual((result['tp'],result['fp'],result['fn']),(1,1,0))
    def test_wrong_class_is_false_positive_and_false_negative(self):
        result=match_detections([Detection((0,0,10,10),1,0.9)],[Annotation((0,0,10,10),0,1)])
        self.assertEqual((result['tp'],result['fp'],result['fn']),(0,1,1))
    def test_empty_detection_sets(self):
        result=match_detections([],[])
        self.assertEqual(result['precision'],0);self.assertEqual(result['recall'],0)
if __name__=='__main__':unittest.main()
