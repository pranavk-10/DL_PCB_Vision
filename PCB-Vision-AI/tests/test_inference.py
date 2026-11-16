"""Inference contracts and rejection paths using explicit synthetic test inputs."""
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ('tensorflow','numpy','cv2','PIL'))

@unittest.skipUnless(AVAILABLE,'TensorFlow/NumPy/OpenCV/Pillow are not installed')
class InferenceTests(unittest.TestCase):
    def setUp(self):
        import numpy as np
        from src import inference
        self.np,self.module=np,inference
    def test_invalid_probability_shapes_nan_and_sum(self):
        for probabilities in (self.np.zeros((1,6)),self.np.full((1,7),self.np.nan),self.np.ones((1,7)),self.np.full((1,7),-1)):
            with self.subTest(shape=probabilities.shape),self.assertRaises(ValueError):self.module.validate_probabilities(probabilities,1)
    def test_valid_probabilities(self):
        self.assertEqual(self.module.validate_probabilities(self.np.full((3,7),1/7),3).shape,(3,7))
    def test_invalid_empty_and_corrupt_upload(self):
        for data in (b'',b'not a PCB image'):
            with self.subTest(data=data),self.assertRaises(ValueError):self.module.decode_uploaded_image(data)
    def test_upload_dimensions_and_rgb(self):
        from PIL import Image
        output=io.BytesIO();Image.new('RGB',(40,30),(255,0,0)).save(output,format='PNG')
        image=self.module.decode_uploaded_image(output.getvalue())
        self.assertEqual(image.shape,(30,40,3))
        with self.assertRaises(ValueError):self.module.decode_uploaded_image(output.getvalue(),max_pixels=100)
    def test_missing_model_is_not_replaced(self):
        with tempfile.TemporaryDirectory(prefix='pcb_missing_model_') as directory:
            with self.assertRaises(FileNotFoundError):self.module.Predictor(Path(directory)/'best.keras')
    def test_incomplete_training_artifact_rejected(self):
        with tempfile.TemporaryDirectory(prefix='pcb_bad_artifact_') as directory:
            path=Path(directory)/'best.keras';path.write_bytes(b'invalid fixture')
            (path.parent/'metadata.json').write_text(json.dumps({'status':'training'}))
            with self.assertRaisesRegex(ValueError,'completed'):self.module.Predictor(path)
    def test_patch_and_batch_contract_on_untrained_test_network(self):
        from config.config import CONFIG
        from src.model import build_cnn
        # Test-only construction verifies shape/math, never used by the application loader.
        predictor=object.__new__(self.module.Predictor);predictor.model=build_cnn(augmentation=False);predictor.settings=CONFIG
        result=predictor.predict_patch(self.np.zeros((40,60),dtype=self.np.uint8))
        self.assertEqual(len(result.probabilities),7);self.assertAlmostEqual(sum(result.probabilities),1,places=5)
        for batch in (self.np.zeros((1,128,128)),self.np.full((1,128,128,1),self.np.nan),self.np.ones((1,128,128,1))*2):
            with self.subTest(shape=batch.shape),self.assertRaises(ValueError):predictor.predict_batch(batch)
if __name__=='__main__':unittest.main()
