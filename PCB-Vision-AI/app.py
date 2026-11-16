"""PCB Vision AI Streamlit inspection workstation; every displayed result is an artifact."""
from __future__ import annotations
from dataclasses import replace
import hashlib
import io
import json
from pathlib import Path
import pandas as pd
from PIL import Image
import streamlit as st
from config.config import CLASS_NAMES, CONFIG

ROOT=CONFIG.root
PAGES=('Home / Dashboard','PCB Image Inspection','Model Performance','Model Information','About')
DESCRIPTIONS=('Break in a conducting trace','Unwanted connection between traces','Notch at a conductor edge',
              'Unwanted protrusion from a trace','Hole inside a conducting region','Unwanted isolated copper',
              'Patch outside annotated defect regions')

def read_json(path: Path):
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding='utf-8'))
    except (OSError,json.JSONDecodeError) as error:
        st.error(f'Could not read {path.name}: {error}')
        return None

def trained_runs() -> dict:
    runs={}
    for path in sorted(CONFIG.models_dir.glob('*/metadata.json')):
        metadata=read_json(path)
        if metadata and metadata.get('status')=='trained' and (path.parent/'best.keras').is_file():
            runs[path.parent.name]=metadata
    return runs

@st.cache_resource(show_spinner='Loading trained model...')
def cached_predictor(model_path: str, model_modified: int, metadata_modified: int):
    from src.inference import Predictor
    return Predictor(model_path)

def change_page(page: str) -> None:
    st.session_state['page']=page

def model_selector(runs: dict, key: str):
    if not runs:
        st.info('No completed training run is available. Prepare the dataset and run python -m src.train. No predictions or performance figures are shown until a real model exists.')
        return None,None
    names=list(runs)
    default=names.index('cnn_adam_aug') if 'cnn_adam_aug' in names else 0
    name=st.selectbox('Training run',names,index=default,key=key)
    return name,runs[name]

def home_page(runs: dict) -> None:
    st.title('Inspection workstation')
    st.write('Classify PCB defects with a CNN trained from scratch, then inspect suspicious image regions.')
    st.button('Inspect a PCB image',type='primary',on_click=change_page,args=('PCB Image Inspection',))
    preparation=read_json(CONFIG.processed_dir/'metadata.json')
    primary=runs.get('cnn_adam_aug')
    evaluation=read_json(CONFIG.metrics_dir/'cnn_adam_aug'/'evaluation.json') if primary else None
    columns=st.columns(3)
    columns[0].metric('Prepared patches',f"{preparation['patch_count']:,}" if preparation else '?')
    columns[1].metric('Completed training runs',len(runs))
    columns[2].metric('Primary test accuracy',f"{evaluation['splits']['test']['accuracy']:.1%}" if evaluation and evaluation.get('model_sha256')==primary.get('model_sha256') else '?')
    st.subheader('From a board to an inspection')
    st.image(str(ROOT/'assets'/'pipeline.svg'),use_container_width=True)
    left,right=st.columns([3,2],gap='large')
    with left:
        st.subheader('Inspection classes')
        st.dataframe(pd.DataFrame({'Class':CLASS_NAMES,'PCB characteristic':DESCRIPTIONS}),hide_index=True,use_container_width=True)
    with right:
        st.subheader('Project readiness')
        st.write('Dataset: prepared' if preparation else 'Dataset: preparation not completed')
        st.write('Primary CNN: trained' if primary else 'Primary CNN: not trained')
        st.write('Evaluation: available' if evaluation else 'Evaluation: not run')
        st.write('Localization: experimental sliding windows, class-aware NMS, and IoU matching.')
        st.write('The classifier learns image patches. A window around a suspicious region is not an exact defect boundary.')
        with st.expander('Prepare and train'):
            st.code('python scripts/download_dataset.py\npython -m src.dataset_loader\npython -m src.preprocessing --confirm-schema\npython -m src.train\npython -m src.evaluate',language='powershell')


def inspection_page(runs: dict) -> None:
    st.title('PCB image inspection')
    st.write('Upload a defect patch for classification or a full board for experimental region inspection.')
    name,metadata=model_selector(runs,'inspection_run')
    uploaded=st.file_uploader('PCB image',type=['jpg','jpeg','png','bmp','tif','tiff','webp'])
    mode=st.radio('Inspection scope',('Full board: sliding windows','Single patch: classification'),horizontal=True)
    with st.expander('Inspection settings',expanded=False):
        controls=st.columns(3)
        window=controls[0].number_input('Window size (pixels)',32,512,CONFIG.window_size,32)
        stride=controls[1].number_input('Stride (pixels)',8,512,CONFIG.stride,8)
        threshold=controls[2].slider('Defect confidence threshold',0.05,0.99,CONFIG.confidence_threshold,0.01)
    if uploaded is None:
        st.info('Upload a PCB image to begin. Results come from the selected completed training run.')
        return
    try:
        from src.inference import decode_uploaded_image
        image=decode_uploaded_image(uploaded.getvalue())
    except (ImportError,OSError,ValueError) as error:
        st.error(f'Cannot open the image: {error}')
        return
    st.image(image,caption=f'Original image ? {image.shape[1]} ? {image.shape[0]} pixels',use_container_width=True)
    enabled=name is not None and stride<=window
    if stride>window:
        st.error('Stride must not exceed window size; otherwise regions would be skipped.')
    image_key=hashlib.sha256(uploaded.getvalue()).hexdigest()
    result_key=(name,image_key,mode,int(window),int(stride),float(threshold))
    if st.button('Run inspection',type='primary',disabled=not enabled):
        try:
            model_path=CONFIG.models_dir/name/'best.keras'
            predictor=cached_predictor(str(model_path),model_path.stat().st_mtime_ns,(model_path.parent/'metadata.json').stat().st_mtime_ns)
            with st.spinner('Inspecting the PCB image...'):
                if mode.startswith('Single'):
                    prediction=predictor.predict_patch(image)
                    st.session_state['inspection_result']={'key':result_key,'prediction':prediction}
                else:
                    from src.localization import draw_detections,inspect_image
                    settings=replace(CONFIG,window_size=int(window),stride=int(stride),confidence_threshold=threshold)
                    inspection=inspect_image(image,predictor,settings)
                    output=io.BytesIO();draw_detections(image,inspection.detections).save(output,format='PNG')
                    st.session_state['inspection_result']={'key':result_key,'inspection':inspection,'annotated_png':output.getvalue()}
        except (ImportError,OSError,ValueError,RuntimeError) as error:
            st.error(f'Inspection failed: {error}. Verify the environment, training metadata, and selected model.')
    result=st.session_state.get('inspection_result')
    if not result or result['key']!=result_key:
        return
    if 'prediction' in result:
        prediction=result['prediction']
        a,b=st.columns(2)
        a.metric('Predicted patch class',prediction.class_name)
        b.metric('Softmax confidence',f'{prediction.confidence:.1%}')
        distribution=pd.DataFrame({'Class':CLASS_NAMES,'Probability':prediction.probabilities}).set_index('Class')
        st.bar_chart(distribution,color='#15836b',horizontal=True)
        st.caption('Softmax confidence is a model score, not a calibrated guarantee of correctness.')
    else:
        inspection=result['inspection']
        a,b,c=st.columns(3)
        a.metric('Suspicious regions',len(inspection.detections));b.metric('Windows evaluated',inspection.window_count);c.metric('Inspection time',f'{inspection.elapsed_seconds:.2f} s')
        st.image(result['annotated_png'],caption='Predicted window regions after class-aware NMS',use_container_width=True)
        if inspection.detections:
            st.dataframe(pd.DataFrame([{'Class':CLASS_NAMES[item.class_id],'Confidence':item.confidence,'x1':item.box[0],'y1':item.box[1],'x2':item.box[2],'y2':item.box[3]} for item in inspection.detections]),hide_index=True,use_container_width=True)
        else:
            st.info('No windows exceeded the selected threshold. This does not certify a defect-free board.')
        st.download_button('Download annotated image',result['annotated_png'],file_name='pcb_inspection.png',mime='image/png')
        st.caption('Experimental localization: these are classified window regions, not trained object-detector boxes. Tiny defects may be missed because training crops and full-board windows have different context.')


def performance_page(runs: dict) -> None:
    st.title('Model performance')
    name,metadata=model_selector(runs,'performance_run')
    if name is None:
        return
    evaluation=read_json(CONFIG.metrics_dir/name/'evaluation.json')
    if not evaluation:
        st.info(f'Evaluation has not been run for {name}. Run python -m src.evaluate --model models/{name}/best.keras.')
        return
    if evaluation.get('model_sha256')!=metadata.get('model_sha256') or evaluation.get('manifest_sha256')!=metadata.get('manifest_sha256'):
        st.error('Evaluation provenance does not match this training artifact. Run evaluation again.')
        return
    test=evaluation['splits']['test'];metrics=st.columns(4)
    for column,label,value in zip(metrics,('Test accuracy','Macro precision','Macro recall','Macro F1'),(test['accuracy'],test['macro']['precision'],test['macro']['recall'],test['macro']['f1'])):
        column.metric(label,f'{value:.2%}')
    st.caption(f"Held-out source groups ? {test['sample_count']:,} test patches ? run {name}")
    history=read_json(CONFIG.metrics_dir/name/'history.json')
    if history:
        left,right=st.columns(2)
        with left:
            st.subheader('Accuracy by epoch');st.line_chart(pd.DataFrame({'Training':history['accuracy'],'Validation':history['val_accuracy']},index=range(1,len(history['accuracy'])+1)),color=['#15836b','#597b92'])
        with right:
            st.subheader('Loss by epoch');st.line_chart(pd.DataFrame({'Training':history['loss'],'Validation':history['val_loss']},index=range(1,len(history['loss'])+1)),color=['#15836b','#597b92'])
    left,right=st.columns([1,1])
    with left:
        st.subheader('Confusion matrix')
        figure=CONFIG.figures_dir/name/'confusion_matrix.png'
        if figure.is_file():
            st.image(str(figure),use_container_width=True)
        else:
            st.dataframe(pd.DataFrame(test['confusion_matrix'],index=CLASS_NAMES,columns=CLASS_NAMES),use_container_width=True)
    with right:
        st.subheader('Per-class results')
        st.dataframe(pd.DataFrame({class_name:test['classification_report'][class_name] for class_name in CLASS_NAMES}).T,use_container_width=True)
        st.write('Macro averages give each class equal weight. Weighted averages reflect class support.')
        st.json(test['weighted'],expanded=False)
    comparison=CONFIG.metrics_dir/'experiments'/'comparison.csv'
    if comparison.is_file():
        st.subheader('Executed experiments');st.dataframe(pd.read_csv(comparison),hide_index=True,use_container_width=True)
    localization=read_json(CONFIG.metrics_dir/name/'localization_test.json')
    if localization and localization.get('model_sha256')==metadata.get('model_sha256'):
        st.subheader('Experimental localization');st.json({key:localization[key] for key in ('image_count','precision','recall','mean_matched_iou')})


def information_page(runs: dict) -> None:
    st.title('Model information')
    st.write('A custom CNN learns local PCB features with shared convolution weights. Every weight starts from random initialization; no pretrained classifier is used.')
    layers=[('Input','128 ? 128 ? 1',0),('Conv 3?3 / ReLU, 32 filters','128 ? 128 ? 32',320),('Max pooling 2?2','64 ? 64 ? 32',0),
            ('Conv 3?3 / ReLU, 64 filters','64 ? 64 ? 64',18496),('Max pooling 2?2','32 ? 32 ? 64',0),
            ('Conv 3?3 / ReLU, 128 filters','32 ? 32 ? 128',73856),('Max pooling 2?2','16 ? 16 ? 128',0),
            ('Flatten','32768',0),('Dense / ReLU','128',4194432),('Dropout 0.3','128',0),('Dense / Softmax','7',903)]
    st.dataframe(pd.DataFrame(layers,columns=['Layer','Output shape','Analytical parameters']),hide_index=True,use_container_width=True)
    st.caption('4,288,007 parameters by architecture arithmetic; training records the actual Keras count. Same padding preserves convolution dimensions. Mild augmentation is active only during training.')
    name,metadata=model_selector(runs,'information_run')
    if metadata:
        st.subheader('Recorded training configuration');st.json(metadata['settings'])
        architecture=CONFIG.figures_dir/name/'architecture.png'
        if architecture.is_file():
            st.image(str(architecture),use_container_width=True)
    else:
        with st.expander('Configured training defaults'):
            st.json(CONFIG.to_dict())
    st.subheader('Dataset and scope')
    st.write('DeepPCB contains aligned template/test images and six annotated defect categories. This project generates a seventh normal class outside the annotated regions and splits PCB board groups before extracting patches.')
    preparation=read_json(CONFIG.processed_dir/'metadata.json')
    if preparation:
        st.json({'patch_count':preparation['patch_count'],'groups_by_split':preparation['groups_by_split'],'counts_by_split':preparation['counts_by_split']})
    st.subheader('Limitations')
    st.write('Normal labels assume complete annotations. Small crops can lose electrical context. Full-board sliding windows change scale and context. Scores are not calibrated. This academic implementation does not establish industrial inspection reliability.')


def about_page() -> None:
    st.title('About PCB Vision AI')
    st.write('An Electronics and Telecommunication Engineering project connecting PCB inspection to neural-network fundamentals: convolutions, artificial neurons, activations, loss, backpropagation, and optimization.')
    st.subheader('Engineering objectives')
    st.write('Train a custom seven-class CNN from scratch; compare flattened ANN and CNN models; compare Adam and SGD; study augmentation and overfitting; evaluate class-wise performance and experimental localization.')
    st.subheader('Technologies')
    st.write('Python ? TensorFlow / Keras ? OpenCV ? NumPy ? Pandas ? Matplotlib ? Scikit-learn ? Streamlit ? Jupyter')
    st.subheader('Academic material')
    for title,path in [('Project report','PROJECT_REPORT.md'),('Deep learning theory','DEEP_LEARNING_THEORY.md'),('Viva preparation','VIVA_QUESTIONS.md')]:
        document=ROOT/'docs'/path
        if document.is_file():
            st.download_button(f'Download {title.lower()}',document.read_text(encoding='utf-8'),file_name=path,mime='text/markdown')
    st.subheader('Future improvements')
    st.write('Collect additional board designs; compare reference/test differences; calibrate probabilities using validation data; study repeated-seed uncertainty; evaluate trained object detection separately from patch classification.')
    st.markdown('[Official DeepPCB dataset](https://github.com/tangsanli5201/DeepPCB)')


def main() -> None:
    st.set_page_config(page_title='PCB Vision AI',page_icon=str(ROOT/'assets'/'pcb_mark.svg'),layout='wide')
    st.markdown('<style>'+(ROOT/'assets'/'styles.css').read_text()+'</style>',unsafe_allow_html=True)
    with st.sidebar:
        st.markdown('<div class="brand">'+(ROOT/'assets'/'pcb_mark.svg').read_text()+'<span>PCB Vision AI</span></div>',unsafe_allow_html=True)
        st.markdown('<div class="scope">Academic PCB inspection</div>',unsafe_allow_html=True)
        page=st.radio('Navigation',PAGES,key='page',label_visibility='collapsed')
        st.caption('Custom CNN ? Seven patch classes')
    runs=trained_runs()
    pages={PAGES[0]:lambda:home_page(runs),PAGES[1]:lambda:inspection_page(runs),PAGES[2]:lambda:performance_page(runs),
           PAGES[3]:lambda:information_page(runs),PAGES[4]:about_page}
    pages[page]()
if __name__=='__main__':
    main()
