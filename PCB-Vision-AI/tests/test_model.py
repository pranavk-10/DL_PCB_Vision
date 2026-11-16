"""Actual TensorFlow operations on synthetic tensors; no academic results are implied."""
import importlib.util
from pathlib import Path
import tempfile
import unittest
AVAILABLE=importlib.util.find_spec('tensorflow') is not None

@unittest.skipUnless(AVAILABLE,'TensorFlow is not installed in the project venv')
class ModelTests(unittest.TestCase):
    def setUp(self):
        import tensorflow as tf
        from src import model
        self.tf,self.module=tf,model
    def tearDown(self):
        self.tf.keras.backend.clear_session()
    def test_cnn_dimensions_and_parameter_count(self):
        model=self.module.build_cnn(augmentation=False)
        self.assertEqual(model.input_shape,(None,128,128,1));self.assertEqual(model.output_shape,(None,7))
        self.assertEqual(model.get_layer('pool_3').output.shape[1:],(16,16,128))
        self.assertEqual(model.count_params(),4288007)
    def test_softmax_is_finite_normalized_and_seven_dimensional(self):
        model=self.module.build_cnn(augmentation=False)
        probability=model(self.tf.ones((2,128,128,1))*0.5,training=False).numpy()
        self.assertEqual(probability.shape,(2,7))
        self.assertTrue(self.tf.reduce_all(self.tf.math.is_finite(probability)))
        self.assertTrue(self.tf.reduce_all(self.tf.abs(self.tf.reduce_sum(probability,axis=1)-1)<1e-5))
    def test_ann_parameter_comparable_and_output_shape(self):
        model=self.module.build_ann()
        self.assertEqual(model.count_params(),4228359);self.assertEqual(model.output_shape,(None,7))
    def test_backpropagation_updates_weights(self):
        model=self.module.compile_model(self.module.build_cnn(augmentation=False))
        before=model.trainable_weights[-2].numpy().copy()
        loss=model.train_on_batch(self.tf.ones((2,128,128,1))*0.5,self.tf.constant([0,1]))
        self.assertTrue(self.tf.reduce_any(model.trainable_weights[-2]!=before))
        self.assertTrue(self.tf.reduce_all(self.tf.math.is_finite(loss)))
    def test_adam_and_sgd_use_identical_initial_cnn_weights(self):
        from dataclasses import replace
        from config.config import CONFIG
        first=self.module.build_cnn(replace(CONFIG,optimizer='adam'),False)
        second=self.module.build_cnn(replace(CONFIG,optimizer='sgd'),False)
        for a,b in zip(first.get_weights(),second.get_weights()):
            self.assertTrue(self.tf.reduce_all(a==b))
    def test_augmentation_inactive_during_inference(self):
        model=self.module.build_cnn(augmentation=True)
        image=self.tf.ones((1,128,128,1))*0.5
        self.assertTrue(self.tf.reduce_all(model(image,training=False)==model(image,training=False)))
    def test_save_load_roundtrip_and_custom_metrics(self):
        model=self.module.compile_model(self.module.build_cnn(augmentation=False))
        image=self.tf.ones((1,128,128,1))*0.5
        with tempfile.TemporaryDirectory(prefix='pcb_model_test_') as directory:
            path=Path(directory)/'fixture.keras';model.save(path)
            loaded=self.module.load_keras_model(path,compile=True)
            self.assertTrue(self.tf.reduce_all(self.tf.abs(model(image)-loaded(image))<1e-6))
    def test_sparse_macro_metrics_use_argmax_not_binary_thresholds(self):
        precision=self.module.SparseMacroPrecision();recall=self.module.SparseMacroRecall()
        labels=self.tf.constant([0,1]);prediction=self.tf.one_hot([0,0],7)
        precision.update_state(labels,prediction);recall.update_state(labels,prediction)
        self.assertAlmostEqual(float(precision.result()),1/14,places=6)
        self.assertAlmostEqual(float(recall.result()),1/7,places=6)
if __name__=='__main__':unittest.main()
