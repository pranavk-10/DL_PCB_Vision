"""Experimental sliding-window localization; windows are not object-detector boxes."""
from __future__ import annotations
import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
import time
from typing import TYPE_CHECKING
from config.config import CLASS_NAMES, CONFIG, Settings
if TYPE_CHECKING:
    from src.inference import Predictor

@dataclass(frozen=True)
class Detection:
    box: tuple[int, int, int, int]
    class_id: int
    confidence: float

@dataclass(frozen=True)
class InspectionResult:
    detections: tuple[Detection, ...]
    window_count: int
    elapsed_seconds: float

def box_iou(first, second) -> float:
    if first[2]<=first[0] or first[3]<=first[1] or second[2]<=second[0] or second[3]<=second[1]:
        raise ValueError('IoU requires positive-area boxes.')
    overlap=max(0,min(first[2],second[2])-max(first[0],second[0]))*max(0,min(first[3],second[3])-max(first[1],second[1]))
    union=(first[2]-first[0])*(first[3]-first[1])+(second[2]-second[0])*(second[3]-second[1])-overlap
    return overlap/union

def non_max_suppression(detections: list[Detection], threshold: float = CONFIG.nms_iou_threshold) -> list[Detection]:
    if not 0<=threshold<=1:
        raise ValueError('NMS threshold must be within [0,1].')
    selected=[]
    for detection in sorted(detections,key=lambda item:(-item.confidence,item.class_id,item.box)):
        if not 0<=detection.class_id<6 or not 0<=detection.confidence<=1:
            raise ValueError('Only defect IDs 0..5 and finite probabilities in [0,1] are supported.')
        if all(other.class_id!=detection.class_id or box_iou(detection.box,other.box)<=threshold for other in selected):
            selected.append(detection)
    return selected

def window_positions(length: int, window: int, stride: int) -> list[int]:
    if length<=0 or window<=0 or not 1<=stride<=window:
        raise ValueError('Positive dimensions and stride within window size are required.')
    if length<=window:
        return [0]
    positions=list(range(0,length-window+1,stride))
    if positions[-1]!=length-window:
        positions.append(length-window)
    return positions

def inspect_image(image, predictor: Predictor, settings: Settings = CONFIG) -> InspectionResult:
    """Stream bounded batches; include image edges, threshold defects, then class-aware NMS."""
    import numpy as np
    from src.preprocessing import preprocess_image
    settings.validate()
    image=np.asarray(image)
    if image.ndim not in (2,3) or image.size==0 or not np.isfinite(image).all():
        raise ValueError('Inspection requires a finite nonempty image.')
    height,width=image.shape[:2]
    if height*width>settings.max_image_pixels:
        raise ValueError('Image exceeds the configured pixel limit.')
    xs=window_positions(width,settings.window_size,settings.stride)
    ys=window_positions(height,settings.window_size,settings.stride)
    window_count=len(xs)*len(ys)
    if window_count>4096:
        raise ValueError(f'{window_count} windows required; increase window size/stride to stay below 4096.')
    start=time.perf_counter()
    candidates,batch,coordinates=[],[],[]
    def flush():
        if not batch:
            return
        probabilities=predictor.predict_batch(np.stack(batch))
        for box,distribution in zip(coordinates,probabilities):
            class_id=int(distribution.argmax());confidence=float(distribution[class_id])
            if class_id!=6 and confidence>=settings.confidence_threshold:
                candidates.append(Detection(box,class_id,confidence))
        batch.clear();coordinates.clear()
    for y in ys:
        for x in xs:
            box=(x,y,min(x+settings.window_size,width),min(y+settings.window_size,height))
            batch.append(preprocess_image(image[y:box[3],x:box[2]],settings.image_size))
            coordinates.append(box)
            if len(batch)==settings.batch_size:
                flush()
    flush()
    return InspectionResult(tuple(non_max_suppression(candidates,settings.nms_iou_threshold)),window_count,time.perf_counter()-start)

def draw_detections(image, detections):
    import numpy as np
    from PIL import Image, ImageDraw
    canvas=Image.fromarray(np.asarray(image).astype(np.uint8)).convert('RGB')
    drawing=ImageDraw.Draw(canvas)
    for detection in detections:
        x1,y1,x2,y2=detection.box;color='#15836b'
        drawing.rectangle((x1,y1,x2-1,y2-1),outline=color,width=2)
        label=f'{CLASS_NAMES[detection.class_id]} {detection.confidence:.1%}'
        bounds=drawing.textbbox((0,0),label)
        text_width,text_height=bounds[2]-bounds[0],bounds[3]-bounds[1]
        tx=max(0,min(x1,canvas.width-text_width-4));ty=max(0,y1-text_height-6)
        drawing.rectangle((tx,ty,min(canvas.width-1,tx+text_width+4),ty+text_height+5),fill=color)
        drawing.text((tx+2,ty),label,fill='white')
    return canvas

def match_detections(predictions: list[Detection], annotations, iou_threshold: float = CONFIG.localization_iou_threshold) -> dict:
    """Greedy class-aware one-to-one matching, without invented background true negatives."""
    if not 0<iou_threshold<=1:
        raise ValueError('Matching threshold must be within (0,1].')
    used,matched_ious=set(),[]
    per_class={str(class_id):{'tp':0,'fp':0,'fn':0} for class_id in range(6)}
    for prediction in sorted(predictions,key=lambda item:-item.confidence):
        if prediction.class_id not in range(6):
            raise ValueError('Localization predictions must be defect classes 0..5.')
        candidates=[(index,box_iou(prediction.box,annotation.box)) for index,annotation in enumerate(annotations)
                    if index not in used and annotation.class_id==prediction.class_id]
        best=max(candidates,key=lambda item:item[1],default=None)
        if best is not None and best[1]>=iou_threshold:
            used.add(best[0]);matched_ious.append(best[1]);per_class[str(prediction.class_id)]['tp']+=1
        else:
            per_class[str(prediction.class_id)]['fp']+=1
    for index,annotation in enumerate(annotations):
        if index not in used:
            per_class[str(annotation.class_id)]['fn']+=1
    tp,fp,fn=(sum(value[key] for value in per_class.values()) for key in ('tp','fp','fn'))
    return {'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.0,'recall':tp/(tp+fn) if tp+fn else 0.0,
            'matched_ious':matched_ious,'mean_matched_iou':sum(matched_ious)/len(matched_ious) if matched_ious else 0.0,'per_class':per_class}

def evaluate_localization(predictor: Predictor, settings: Settings = CONFIG, split: str = 'test') -> dict:
    from src.dataset_loader import parse_annotations,validate_box
    from src.preprocessing import load_manifest,read_image
    if split not in {'validation','test'}:
        raise ValueError('Localization evaluation uses validation or test images.')
    frame,preparation=load_manifest(settings)
    if predictor.metadata['manifest_sha256']!=preparation['manifest_sha256']:
        raise ValueError("Localization must use the model's original source split.")
    images=frame[frame['split']==split].drop_duplicates('image_id')
    if images.empty:
        raise ValueError(f'No source images in {split}.')
    matches,records=[],[]
    for row in images.itertuples():
        image=read_image(row.source_image,settings.max_image_pixels)
        annotations=parse_annotations(row.annotation_path,preparation['settings']['annotation_origin'])
        for annotation in annotations:
            validate_box(annotation.box,image.shape[1],image.shape[0])
        inspection=inspect_image(image,predictor,settings)
        match=match_detections(list(inspection.detections),annotations,settings.localization_iou_threshold)
        matches.append(match)
        records.append({'image_id':row.image_id,'window_count':inspection.window_count,'elapsed_seconds':inspection.elapsed_seconds,
                        'detections':[asdict(item) for item in inspection.detections],**match})
    tp,fp,fn=(sum(match[key] for match in matches) for key in ('tp','fp','fn'))
    ious=[value for match in matches for value in match['matched_ious']]
    per_class={}
    for class_id in range(6):
        values={key:sum(match['per_class'][str(class_id)][key] for match in matches) for key in ('tp','fp','fn')}
        values['precision']=values['tp']/(values['tp']+values['fp']) if values['tp']+values['fp'] else 0.0
        values['recall']=values['tp']/(values['tp']+values['fn']) if values['tp']+values['fn'] else 0.0
        per_class[CLASS_NAMES[class_id]]=values
    return {'scope':'experimental window localization; not detector mAP','split':split,'image_count':len(records),
            'model_sha256':predictor.metadata['model_sha256'],'manifest_sha256':preparation['manifest_sha256'],'settings':settings.to_dict(),
            'tp':tp,'fp':fp,'fn':fn,'precision':tp/(tp+fp) if tp+fp else 0.0,'recall':tp/(tp+fn) if tp+fn else 0.0,
            'mean_matched_iou':sum(ious)/len(ious) if ious else 0.0,'per_class':per_class,'images':records}

def main() -> None:
    from dataclasses import replace
    from src.inference import Predictor,decode_uploaded_image
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--image',type=Path)
    parser.add_argument('--model',type=Path,default=CONFIG.model_path)
    parser.add_argument('--window',type=int,default=CONFIG.window_size)
    parser.add_argument('--stride',type=int,default=CONFIG.stride)
    parser.add_argument('--threshold',type=float,default=CONFIG.confidence_threshold)
    parser.add_argument('--evaluate',action='store_true')
    parser.add_argument('--split',choices=('validation','test'),default='test')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args();logging.basicConfig(level=logging.INFO)
    settings=replace(CONFIG,window_size=args.window,stride=args.stride,confidence_threshold=args.threshold)
    try:
        predictor=Predictor(args.model,settings)
        if args.evaluate:
            result=evaluate_localization(predictor,settings,args.split)
            output=args.output or settings.metrics_dir/predictor.path.parent.name/f'localization_{args.split}.json'
            output.parent.mkdir(parents=True,exist_ok=True);output.write_text(json.dumps(result,indent=2),encoding='utf-8')
        elif args.image:
            image=decode_uploaded_image(args.image.read_bytes());inspection=inspect_image(image,predictor,settings)
            output=args.output or settings.figures_dir/'inspection.png'
            output.parent.mkdir(parents=True,exist_ok=True);draw_detections(image,inspection.detections).save(output,format='PNG')
            logging.info('%s suspicious regions; %s windows inspected.',len(inspection.detections),inspection.window_count)
        else:
            raise ValueError('Supply --image PATH or --evaluate.')
    except (OSError,ValueError,RuntimeError) as error:
        parser.exit(1,f'Localization failed: {error}\n')
if __name__=='__main__':
    main()
