"""Evaluate validation-selected checkpoints using seven explicit class labels."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from config.config import CLASS_NAMES, CONFIG, Settings
from src.inference import Predictor, validate_probabilities
from src.preprocessing import load_manifest, make_tf_dataset
LOGGER = logging.getLogger(__name__)

def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict:
    labels = np.asarray(labels, dtype=int).reshape(-1)
    if len(labels) == 0 or not np.isin(labels, range(7)).all():
        raise ValueError("Evaluation requires nonempty labels within 0..6.")
    probabilities = validate_probabilities(probabilities, len(labels))
    predicted = probabilities.argmax(axis=1)
    report = classification_report(labels, predicted, labels=list(range(7)), target_names=list(CLASS_NAMES), zero_division=0, output_dict=True)
    result = {"sample_count": len(labels), "accuracy": float(accuracy_score(labels, predicted)), "classification_report": report,
              "confusion_matrix": confusion_matrix(labels, predicted, labels=list(range(7))).tolist()}
    for average in ("macro", "weighted"):
        precision, recall, f1, _ = precision_recall_fscore_support(labels, predicted, labels=list(range(7)), average=average, zero_division=0)
        result[average] = {"precision": float(precision), "recall": float(recall), "f1": float(f1)}
    return result

def plot_training_history(history: dict, output_dir: Path) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for metric in ("accuracy", "loss"):
        if metric not in history or f"val_{metric}" not in history:
            raise ValueError(f"History lacks training/validation {metric}.")
        figure, axis = plt.subplots(figsize=(8, 4))
        axis.plot(range(1, len(history[metric])+1), history[metric], color="#15836b", label="Training")
        axis.plot(range(1, len(history[f'val_{metric}'])+1), history[f"val_{metric}"], color="#597b92", label="Validation")
        axis.set(xlabel="Epoch", ylabel=metric.capitalize(), title=f"Training and validation {metric}")
        axis.legend(); figure.tight_layout()
        path = output_dir / f"training_{metric}.png"
        figure.savefig(path, dpi=150); plt.close(figure); outputs.append(path)
    return outputs

def plot_confusion_matrix(matrix: np.ndarray, output: Path) -> Path:
    matrix = np.asarray(matrix)
    figure, axis = plt.subplots(figsize=(10, 8))
    display = axis.imshow(matrix, cmap="Greens", vmin=0)
    figure.colorbar(display, ax=axis, label="Test patches")
    axis.set(xticks=range(7), yticks=range(7), xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
             xlabel="Predicted class", ylabel="True class", title="Held-out patch confusion matrix")
    plt.setp(axis.get_xticklabels(), rotation=35, ha="right")
    threshold = matrix.max()/2 if matrix.size else 0
    for row in range(7):
        for column in range(7):
            axis.text(column, row, str(matrix[row,column]), ha="center", va="center", color="white" if matrix[row,column]>threshold else "#17231e")
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout(); figure.savefig(output,dpi=150); plt.close(figure)
    return output

def plot_per_class(report: dict, output: Path) -> Path:
    positions=np.arange(7)
    figure,axis=plt.subplots(figsize=(11,5))
    axis.bar(positions-0.18,[report[name]['precision'] for name in CLASS_NAMES],0.36,label='Precision',color='#15836b')
    axis.bar(positions+0.18,[report[name]['recall'] for name in CLASS_NAMES],0.36,label='Recall',color='#597b92')
    axis.set(xticks=positions,xticklabels=CLASS_NAMES,ylabel='Score',ylim=(0,1.05),title='Per-class held-out precision and recall')
    plt.setp(axis.get_xticklabels(),rotation=25,ha='right');axis.legend()
    output.parent.mkdir(parents=True,exist_ok=True)
    figure.tight_layout();figure.savefig(output,dpi=150);plt.close(figure)
    return output

def evaluate_model(model_path: str | Path, settings: Settings = CONFIG, run_name: str | None = None) -> dict:
    predictor=Predictor(model_path,settings)
    run_name=run_name or predictor.path.parent.name
    frame,preparation=load_manifest(settings)
    if predictor.metadata['manifest_sha256']!=preparation['manifest_sha256']:
        raise ValueError('Evaluation manifest differs from the original training split.')
    results={'run_name':run_name,'model_sha256':predictor.metadata['model_sha256'],'manifest_sha256':preparation['manifest_sha256'],
             'class_names':list(CLASS_NAMES),'parameter_count':predictor.metadata['parameter_count'],
             'training_seconds':predictor.metadata['training_seconds'],'splits':{}}
    test_predictions=None
    for split_name in ('train','validation','test'):
        subset=frame[frame['split']==split_name]
        if subset.empty:
            raise ValueError(f'{split_name} split is empty.')
        probabilities=np.concatenate([predictor.predict_batch(images.numpy()) for images,_ in make_tf_dataset(subset,settings)])
        labels=subset['class_id'].to_numpy(dtype=int)
        results['splits'][split_name]=classification_metrics(labels,probabilities)
        if split_name=='test':
            test_predictions=subset.copy()
            test_predictions['predicted_class_id']=probabilities.argmax(axis=1)
            test_predictions['confidence']=probabilities.max(axis=1)
            for class_id in range(7):
                test_predictions[f'probability_{class_id}']=probabilities[:,class_id]
    metrics_dir,figures_dir=settings.metrics_dir/run_name,settings.figures_dir/run_name
    metrics_dir.mkdir(parents=True,exist_ok=True)
    (metrics_dir/'evaluation.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    test_predictions.to_csv(metrics_dir/'test_predictions.csv',index=False)
    test_results=results['splits']['test']
    plot_confusion_matrix(np.array(test_results['confusion_matrix']),figures_dir/'confusion_matrix.png')
    plot_per_class(test_results['classification_report'],figures_dir/'per_class_precision_recall.png')
    LOGGER.info('Actual test accuracy %.4f, macro F1 %.4f',test_results['accuracy'],test_results['macro']['f1'])
    return results

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model',type=Path,default=CONFIG.model_path)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    try:
        evaluate_model(args.model)
    except (OSError,ValueError,RuntimeError) as error:
        parser.exit(1,f'Evaluation failed: {error}\n')
if __name__=='__main__':
    main()
