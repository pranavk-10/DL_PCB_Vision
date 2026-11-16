"""Use explicit synthetic arrays to check crops, normalization, normals, and corrupt inputs."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ('numpy','cv2','PIL','pandas'))

@unittest.skipUnless(AVAILABLE,'NumPy/OpenCV/Pillow/Pandas are not installed in the project venv')
class PreprocessingTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        self.np=np
        from src import preprocessing
        self.pre=preprocessing
    def test_output_shape_range_and_dtype(self):
        value=self.pre.preprocess_image(self.np.full((36,47),255,dtype=self.np.uint8))
        self.assertEqual(value.shape,(128,128,1));self.assertEqual(value.dtype,self.np.float32)
        self.assertTrue(self.np.all(value==1))
    def test_zero_one_integer_values_are_not_already_normalized(self):
        value=self.pre.preprocess_image(self.np.ones((128,128),dtype=self.np.uint8))
        self.assertTrue(self.np.allclose(value,1/255))
    def test_float_normalization_not_applied_twice(self):
        image=self.np.full((128,128),0.5,dtype=self.np.float32)
        self.assertTrue(self.np.allclose(self.pre.preprocess_image(image),0.5))
    def test_rgb_grayscale_and_one_channel_equivalence(self):
        gray=self.np.full((128,128),128,dtype=self.np.uint8)
        rgb=self.np.repeat(gray[...,None],3,axis=2)
        self.assertTrue(self.np.allclose(self.pre.preprocess_image(gray),self.pre.preprocess_image(rgb),atol=1e-6))
    def test_padding_clamps_and_crop_is_correct(self):
        image=self.np.zeros((20,20),dtype=self.np.uint8);image[:5,:5]=255
        patch=self.pre.extract_defect_patch(image,(0,0,5,5),padding=0)
        self.assertTrue(self.np.all(patch==1))
        self.assertEqual(self.pre.padded_box((0,0,5,5),20,20,10),(0,0,15,15))
    def test_invalid_arrays_and_bounds_rejected(self):
        for image in (self.np.array([]),self.np.full((3,3),self.np.nan),self.np.full((3,3),256),self.np.zeros((3,3,2))):
            with self.subTest(shape=image.shape),self.assertRaises(ValueError):self.pre.preprocess_image(image)
        with self.assertRaises(ValueError):self.pre.extract_defect_patch(self.np.zeros((10,10)),(0,0,11,10))
    def test_normal_boxes_clear_all_defects_and_are_unique(self):
        boxes=[(25,25,75,75),(100,100,150,150)]
        normal=self.pre.generate_normal_boxes((256,256),boxes,self.np.random.default_rng(42),window_size=32,count=8,clearance=12)
        self.assertEqual(len(set(normal)),len(normal));self.assertTrue(normal)
        for window in normal:
            for defect in boxes:self.assertEqual(self.pre.intersection_area(window,self.pre.padded_box(defect,256,256,12)),0)
    def test_normal_sampling_does_not_invent_samples_when_blocked(self):
        normal=self.pre.generate_normal_boxes((128,128),[(0,0,128,128)],self.np.random.default_rng(42),count=4)
        self.assertEqual(normal,[])
    def test_corrupt_image_rejected(self):
        with tempfile.TemporaryDirectory(prefix='pcb_image_test_') as directory:
            path=Path(directory)/'broken.png';path.write_bytes(b'not an image')
            with self.assertRaises(ValueError):self.pre.read_image(path)
    def test_saved_png_training_and_inference_preprocessing_match(self):
        with tempfile.TemporaryDirectory(prefix='pcb_resize_test_') as directory:
            gray=self.np.arange(30*40,dtype=self.np.uint16).reshape(30,40)%256
            gray=gray.astype(self.np.uint8);path=Path(directory)/'patch.png'
            self.pre._save_gray(path,gray,128)
            stored=self.pre.preprocess_image(self.pre.read_image(path))
            direct=self.pre.preprocess_image(gray)
            self.assertTrue(self.np.allclose(stored,direct,atol=1/255))
if __name__=='__main__':unittest.main()
