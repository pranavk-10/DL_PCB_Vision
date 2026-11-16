"""End-to-end one-epoch smoke test on synthetic patches in a temporary directory only."""
import importlib.util
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
AVAILABLE=all(importlib.util.find_spec(name) is not None for name in ('tensorflow','numpy','cv2','PIL','pandas','matplotlib','sklearn'))

@unittest.skipUnless(AVAILABLE,'Scientific/TensorFlow dependencies are not installed')
class TrainingIntegrationTests(unittest.TestCase):
    def test_training_checkpoint_evaluation_and_inference_roundtrip(self):
        import numpy as np
        import pandas as pd
        from PIL import Image
        from config.config import CONFIG
        from src.train import train_model
        from src.inference import Predictor
        with tempfile.TemporaryDirectory(prefix='pcb_training_smoke_') as directory:
            root=Path(directory)
            settings=replace(CONFIG,processed_dir=root/'processed',models_dir=root/'models',metrics_dir=root/'metrics',figures_dir=root/'figures',epochs=1,batch_size=7,augmentation=False)
            settings.processed_dir.mkdir();rows=[]
            generator=np.random.default_rng(42)
            for split in ('train','validation','test'):
                for class_id in range(7):
                    relative=f'{split}/{class_id}.png';path=settings.processed_dir/relative;path.parent.mkdir(exist_ok=True)
                    Image.fromarray(generator.integers(0,256,size=(128,128),dtype=np.uint8)).save(path)
                    rows.append({'patch_path':relative,'split':split,'class_id':class_id,'image_id':f'{split}_{class_id}',
                                 'group_id':split,'source_image':f'{split}_{class_id}_test.png','template_image':f'{split}_{class_id}_temp.png'})
            manifest=settings.processed_dir/'manifest.csv';pd.DataFrame(rows).to_csv(manifest,index=False)
            metadata={'settings':settings.to_dict(),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),
                      'dataset_fingerprint':'synthetic_unit_test_only','groups_by_split':{name:[name] for name in ('train','validation','test')}}
            (settings.processed_dir/'metadata.json').write_text(json.dumps(metadata))
            trained=train_model(settings,run_name='smoke_test')
            self.assertEqual(trained['status'],'trained');self.assertEqual(trained['epochs_completed'],1)
            predictor=Predictor(settings.models_dir/'smoke_test'/'best.keras',settings)
            probabilities=predictor.predict_patch(np.zeros((128,128),dtype=np.uint8)).probabilities
            self.assertAlmostEqual(sum(probabilities),1,places=5)
            evaluation=json.loads((settings.metrics_dir/'smoke_test'/'evaluation.json').read_text())
            self.assertEqual(evaluation['splits']['test']['sample_count'],7)
            self.assertTrue((settings.figures_dir/'smoke_test'/'confusion_matrix.png').is_file())
if __name__=='__main__':unittest.main()
