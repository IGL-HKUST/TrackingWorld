import sys
import os

# Add the project root to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.join(current_dir, "..", "submodules", "Grounded-SAM-2")  # adjust if needed
sys.path.append(os.path.abspath(project_root))
import argparse
import cv2
import json
import torch
import imageio
import numpy as np
import supervision as sv
from pathlib import Path
from supervision.draw.color import ColorPalette

from utils.supervision_utils import CUSTOM_COLOR_MAP
from PIL import Image
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection
import torchvision 
from tqdm import tqdm

"""
Hyper parameters
"""
parser = argparse.ArgumentParser()
parser.add_argument('--grounding-model', default="IDEA-Research/grounding-dino-base")
parser.add_argument("--workdir", default="./demo_data")
parser.add_argument("--sam2-checkpoint", default="./checkpoints/sam2.1_hiera_large.pt")
parser.add_argument("--sam2-model-config", default="configs/sam2.1/sam2.1_hiera_l.yaml")
parser.add_argument("--output-dir", default="gsam2")
parser.add_argument("--input-dir", default="ram")
parser.add_argument("--force-cpu", action="store_true")
args = parser.parse_args()

GROUNDING_MODEL = args.grounding_model
INPUT_DIR = args.workdir
SAM2_CHECKPOINT = args.sam2_checkpoint
SAM2_MODEL_CONFIG = args.sam2_model_config
DEVICE = "cuda" if torch.cuda.is_available() and not args.force_cpu else "cpu"
OUTPUT_DIR = Path(args.output_dir)


# environment settings
# use bfloat16
torch.autocast(device_type=DEVICE, dtype=torch.bfloat16).__enter__()

if torch.cuda.get_device_properties(0).major >= 8:
    # turn on tfloat32 for Ampere GPUs (https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

# build SAM2 image predictor
sam2_checkpoint = SAM2_CHECKPOINT
model_cfg = SAM2_MODEL_CONFIG
sam2_model = build_sam2(model_cfg, sam2_checkpoint, device=DEVICE)
sam2_predictor = SAM2ImagePredictor(sam2_model)

# build grounding dino from huggingface
model_id = GROUNDING_MODEL
processor = AutoProcessor.from_pretrained(model_id)
grounding_model = AutoModelForZeroShotObjectDetection.from_pretrained(model_id).to(DEVICE)

def id_to_colors(id): # id to color
    rgb = np.zeros((3, ), dtype=np.uint8)
    for i in range(3):
        rgb[i] = id % 256
        id = id // 256
    return rgb

videos = sorted(os.listdir(INPUT_DIR))

for i,video in tqdm(enumerate(videos)):
  #if video in ['tapvid3d_634378055350569306_280_000_300_000_3_qZfnKf-L0RkIHWLqwNAG5Q']:
    work_dir = os.path.join(INPUT_DIR, video)

    image_dir_path = os.path.join(work_dir, "rgb")

    idx_to_id = [i for i in range(256*256*256)]
    np.random.shuffle(idx_to_id) # mapping to randomize idx to id to get random color

    text_prompt_file = os.path.join(work_dir, args.input_dir, "tags.json")
    with open(text_prompt_file, "r") as f:
        dyn_objs = json.load(f)["dynamic"]
    text_input = ". ".join(dyn_objs)
    
    output_path_vis = os.path.join(work_dir, OUTPUT_DIR, "vis")
    output_path_mask = os.path.join(work_dir, OUTPUT_DIR, "mask")
    os.makedirs(output_path_vis, exist_ok=True)
    os.makedirs(output_path_mask, exist_ok=True)

    for image_file in sorted(os.listdir(image_dir_path)):

        if not(image_file.endswith(".jpg") or image_file.endswith(".png")):
            continue

        full_image_path = os.path.join(image_dir_path, image_file)

        image_pil = Image.open(full_image_path).convert("RGB")
        image = np.array(image_pil)

        sam2_predictor.set_image(image)

        inputs = processor(images=image, text=text_input, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            outputs = grounding_model(**inputs)

        results = processor.post_process_grounded_object_detection(
            outputs,
            inputs.input_ids,
            box_threshold=0.3,
            text_threshold=0.3,
            target_sizes=[image_pil.size[::-1]]
        )

        # get the box prompt for SAM 2
        input_boxes = results[0]["boxes"].cpu().numpy()

        if input_boxes.shape[0] != 0:

            masks, scores, logits = sam2_predictor.predict(
                point_coords=None,
                point_labels=None,
                box=input_boxes,
                multimask_output=False,
            )

            """
            Post-process the output of the model to get the masks, scores, and logits for visualization
            """
            # convert the shape to (n, H, W)
            if masks.ndim == 4:
                masks = masks.squeeze(1)

            confidences = results[0]["scores"].cpu().numpy().tolist()
            class_names = results[0]["labels"]
            class_ids = np.array(list(range(len(class_names))))

            """
            Visualize image with supervision useful API
            """
            img = cv2.imread(full_image_path)
            detections = sv.Detections(
                xyxy=input_boxes,  # (n, 4)
                mask=masks.astype(bool),  # (n, h, w)
                class_id=class_ids,
                confidence=np.array(confidences)
            )

            assert(len(detections.class_id) > 0)

            nms_idx = torchvision.ops.nms(
                        torch.from_numpy(detections.xyxy).float(), 
                        torch.from_numpy(detections.confidence).float(), 
                        0.5
                    ).numpy().tolist()

            detections.xyxy = detections.xyxy[nms_idx]
            detections.class_id = detections.class_id[nms_idx]
            detections.confidence = detections.confidence[nms_idx]
            detections.mask = detections.mask[nms_idx]

            labels = [
                f"{class_names[id]} {confidence:.2f}"
                for id, confidence
                in zip(detections.class_id, detections.confidence)
            ]

            box_annotator = sv.BoxAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = box_annotator.annotate(scene=img.copy(), detections=detections)

            label_annotator = sv.LabelAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = label_annotator.annotate(scene=annotated_frame, detections=detections, labels=labels)

            mask_annotator = sv.MaskAnnotator(color=ColorPalette.from_hex(CUSTOM_COLOR_MAP))
            annotated_frame = mask_annotator.annotate(scene=annotated_frame, detections=detections)

            masks = detections.mask
            labels = detections.class_id

            assert(np.sum(labels == -1) == 0) # check if any label == -1? concept graph has a bug with this

            color_mask = np.zeros(image.shape, dtype=np.uint8)

            obj_info_json = []

            #sort masks according to size
            mask_size = [np.sum(mask) for mask in masks]
            sorted_mask_idx = np.argsort(mask_size)[::-1]

            for idx in sorted_mask_idx: # render from largest to smallest
                
                mask = masks[idx]
                color_mask[mask] = id_to_colors(idx_to_id[idx])

                obj_info_json.append({
                    "id": idx_to_id[idx],
                    "label": class_names[labels[idx]],
                    "score": float(detections.confidence[idx]),
                })

            color_mask = cv2.cvtColor(color_mask, cv2.COLOR_RGB2BGR)

            cv2.imwrite(os.path.join(output_path_vis, image_file), annotated_frame) # VISUALIZATION
            image_file = image_file.replace(".jpg", ".png")
            cv2.imwrite(os.path.join(output_path_mask, image_file), color_mask)
            with open(os.path.join(output_path_mask, image_file.replace(".png", ".json")), "w") as f:
                json.dump(obj_info_json, f)
        
        else:
            
            imageio.imwrite(os.path.join(output_path_vis, image_file), image) # VISUALIZATION
            image_file = image_file.replace(".jpg", ".png")
            cv2.imwrite(os.path.join(output_path_mask, image_file), np.zeros(image.shape, dtype=np.uint8))
            with open(os.path.join(output_path_mask, image_file.replace(".png", ".json")), "w") as f:
                json.dump([], f)
