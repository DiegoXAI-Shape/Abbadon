import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
import sys

# Add the path to utils/models to import sam_loader
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'models')))
from sam_loader import load_sam2

def generate_masks(input_root, output_root, device="cuda"):
    """
    Generates SAM masks for images in Cat and Dog folders.
    """
    # Load SAM 2 Predictor
    print(f"Loading SAM 2 on {device}...")
    predictor = load_sam2(device=device)
    
    categories = ['Cat', 'Dog']
    
    for cat in categories:
        input_dir = os.path.join(input_root, cat)
        output_dir = os.path.join(output_root, cat)
        
        if not os.path.exists(input_dir):
            print(f"Directory {input_dir} not found. Skipping.")
            continue
            
        os.makedirs(output_dir, exist_ok=True)
        
        images = [f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        print(f"Found {len(images)} images in {cat} folder.")
        
        for img_name in tqdm(images, desc=f"Processing {cat}"):
            img_path = os.path.join(input_dir, img_name)
            output_path = os.path.join(output_dir, img_name.split('.')[0] + '.png')
            
            if os.path.exists(output_path):
                continue # Skip if already processed
            
            try:
                # Read image
                image = cv2.imread(img_path)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, _ = image.shape
                
                # Set image for predictor
                predictor.set_image(image)
                
                # Use center point as prompt (common for pet images)
                input_point = np.array([[w // 2, h // 2]])
                input_label = np.array([1]) # 1 for foreground
                
                # Predict
                masks, scores, logits = predictor.predict(
                    point_coords=input_point,
                    point_labels=input_label,
                    multimask_output=True
                )
                
                # Pick the mask with the highest score
                best_mask_idx = np.argmax(scores)
                best_mask = masks[best_mask_idx]
                
                # Convert boolean mask to uint8 (0 or 255)
                mask_uint8 = (best_mask.astype(np.uint8)) * 255
                
                # Save mask
                cv2.imwrite(output_path, mask_uint8)
                
            except Exception as e:
                print(f"Error processing {img_path}: {e}")

if __name__ == "__main__":
    # Correct paths based on the project structure
    # The script is in source/utils/scripts/
    # data is in source/data/
    
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    base_data_dir = os.path.abspath(os.path.join(current_script_dir, '..', '..', 'data'))
    
    input_root = os.path.join(base_data_dir, 'PetImages')
    output_root = os.path.join(base_data_dir, 'sam_masks')
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    generate_masks(input_root, output_root, device=device)
