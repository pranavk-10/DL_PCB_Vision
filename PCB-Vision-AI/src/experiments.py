"""Four actual runs support three controlled comparisons; unrun results stay unrun."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
import logging
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from config.config import CONFIG,Settings
from src.evaluate import evaluate_model
from src.preprocessing import load_manifest
from src.train import train_model
EXPERIMENT_RUNS=(('ann_adam_plain','ann','adam',False),('cnn_adam_plain','cnn','adam',False),
                 ('cnn_sgd_plain','cnn','sgd',False),('cnn_adam_aug','cnn','adam',True))
COMPARISONS={'ANN vs CNN':('ann_adam_plain','cnn_adam_plain'),'Adam vs SGD':('cnn_adam_plain','cnn_sgd_plain'),
             'Data augmentation':('cnn_adam_plain','cnn_adam_aug')}

def experiment_plan(settings: Settings=CONFIG) -> dict:
    return {'status':'not_run','seed':settings.seed,'comparisons':COMPARISONS,
            'controlled_conditions':'same source groups, batch size, epoch budget, dropout, class weights, callbacks; both optimizers use the configured learning rate',
            'runs':[{'run_name':name,'architecture':architecture,'status':'not_run',
                     'settings':replace(settings,optimizer=optimizer,augmentation=augmentation).to_dict()}
                    for name,architecture,optimizer,augmentation in EXPERIMENT_RUNS]}

def plot_experiment_comparisons(rows: list[dict], histories: dict, directory: Path) -> None:
    directory.mkdir(parents=True,exist_ok=True);indexed={row['run_name']:row for row in rows}
    for comparison,names in COMPARISONS.items():
        if not all(name in indexed for name in names):
            continue
        selected=[indexed[name] for name in names]
        figure,axes=plt.subplots(1,3,figsize=(15,4))
        axes[0].bar(names,[row['test_accuracy'] for row in selected],color=['#597b92','#15836b'])
        axes[0].set(title='Held-out patch accuracy',ylim=(0,1.05),ylabel='Accuracy')
        axes[1].bar(names,[row['training_seconds'] for row in selected],color=['#597b92','#15836b'])
        axes[1].set(title='Training duration',ylabel='Seconds')
        for name,color in zip(names,('#597b92','#15836b')):
            axes[2].plot(range(1,len(histories[name]['val_loss'])+1),histories[name]['val_loss'],color=color,label=name)
        axes[2].set(title='Validation loss convergence',xlabel='Epoch',ylabel='Loss');axes[2].legend(fontsize=8)
        for axis in axes[:2]:
            axis.tick_params(axis='x',rotation=15)
        figure.suptitle(comparison);figure.tight_layout()
        figure.savefig(directory/(comparison.lower().replace(' ','_')+'.png'),dpi=150);plt.close(figure)
        figure,axes=plt.subplots(1,2,figsize=(12,4))
        for name,color in zip(names,('#597b92','#15836b')):
            for axis,key in zip(axes,('loss','val_loss')):
                axis.plot(range(1,len(histories[name][key])+1),histories[name][key],color=color,label=name)
                axis.set(xlabel='Epoch',ylabel=key,title=key.replace('_',' ').capitalize());axis.legend(fontsize=8)
        figure.tight_layout();figure.savefig(directory/(comparison.lower().replace(' ','_')+'_losses.png'),dpi=150);plt.close(figure)
    figure,axis=plt.subplots(figsize=(10,4))
    axis.bar([row['run_name'] for row in rows],[row['test_macro_f1'] for row in rows],color='#15836b')
    axis.set(ylabel='Macro F1',ylim=(0,1.05),title='Executed model comparison');axis.tick_params(axis='x',rotation=15)
    figure.tight_layout();figure.savefig(directory/'model_comparison.png',dpi=150);plt.close(figure)

def run_experiments(settings: Settings=CONFIG, plan_only: bool=False, overwrite: bool=False) -> dict:
    plan=experiment_plan(settings);directory=settings.metrics_dir/'experiments';directory.mkdir(parents=True,exist_ok=True)
    plan_path=directory/'experiment_status.json';plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    if plan_only:
        return plan
    _,preparation=load_manifest(settings);plan['manifest_sha256']=preparation['manifest_sha256'];rows,histories=[],{}
    for index,(name,architecture,optimizer,augmentation) in enumerate(EXPERIMENT_RUNS):
        run_settings=replace(settings,optimizer=optimizer,augmentation=augmentation)
        metadata_path=settings.models_dir/name/'metadata.json'
        try:
            if metadata_path.is_file() and not overwrite:
                metadata=json.loads(metadata_path.read_text())
                keys=('seed','epochs','batch_size','learning_rate','augmentation','optimizer','dropout','early_stopping_patience','reduce_lr_patience')
                if metadata.get('status')!='trained' or metadata.get('manifest_sha256')!=preparation['manifest_sha256'] or metadata.get('architecture')!=architecture or any(metadata['settings'][key]!=run_settings.to_dict()[key] for key in keys):
                    raise ValueError(f'Existing {name} differs from controlled conditions. Use --overwrite or preserve it elsewhere.')
            else:
                metadata=train_model(run_settings,architecture,name,overwrite,evaluate=False)
            result=evaluate_model(settings.models_dir/name/'best.keras',run_settings,name)
            histories[name]=json.loads((settings.metrics_dir/name/'history.json').read_text())
            row={'run_name':name,'architecture':architecture,'optimizer':optimizer,'augmentation':augmentation,
                 'train_accuracy':result['splits']['train']['accuracy'],'validation_accuracy':result['splits']['validation']['accuracy'],
                 'test_accuracy':result['splits']['test']['accuracy'],'test_macro_f1':result['splits']['test']['macro']['f1'],
                 'training_seconds':metadata['training_seconds'],'parameter_count':metadata['parameter_count'],
                 'epochs_completed':metadata['epochs_completed'],'best_epoch':metadata['best_epoch']}
            rows.append(row);plan['runs'][index]['status']='completed';plan['runs'][index]['measured_results']=row
            pd.DataFrame(rows).to_csv(directory/'comparison.csv',index=False)
            plot_experiment_comparisons(rows,histories,settings.figures_dir/'experiments')
        except Exception as error:
            plan['status']='incomplete';plan['runs'][index]['status']='failed';plan['runs'][index]['error']=f'{type(error).__name__}: {error}'
            plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8');raise
        plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    plan['status']='completed';plan_path.write_text(json.dumps(plan,indent=2),encoding='utf-8')
    return plan

def main() -> None:
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--epochs',type=int,default=CONFIG.epochs);parser.add_argument('--plan-only',action='store_true');parser.add_argument('--overwrite',action='store_true')
    args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    try:
        run_experiments(replace(CONFIG,epochs=args.epochs),args.plan_only,args.overwrite)
    except (OSError,ValueError,RuntimeError) as error:
        parser.exit(1,f'Experiments failed: {error}\n')
if __name__=='__main__':
    main()
