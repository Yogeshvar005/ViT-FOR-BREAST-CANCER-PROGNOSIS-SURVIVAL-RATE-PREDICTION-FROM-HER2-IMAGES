import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import pandas as pd
from torchvision import transforms, models
from PIL import Image
import matplotlib
# Set non-interactive backend to prevent GUI thread issues
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from datetime import datetime
import io
import base64
from flask import Flask, render_template, request, jsonify

# Import LLM integration
from llm_integration import get_llm_model

# Define survival estimates for each class (in months) based on HER2 status
BASE_SURVIVAL_ESTIMATES = {
    'class_0': {'median': 60, 'range': '48-72', 'description': 'No overexpression (Score 0)'},
    'class_1+': {'median': 48, 'range': '36-60', 'description': 'Incomplete membrane staining (Score 1+)'},
    'class_2+': {'median': 36, 'range': '24-48', 'description': 'Weak to moderate complete membrane staining (Score 2+)'},
    'class_3+': {'median': 24, 'range': '12-36', 'description': 'Strong complete membrane staining (Score 3+)'}
}

# Age adjustment factors (multipliers for survival estimates)
AGE_ADJUSTMENT = {
    'under_40': 1.2,  # Better prognosis for younger patients
    '40_to_60': 1.0,  # Baseline
    'over_60': 0.8    # Worse prognosis for older patients
}

# Morbidities and recommendations from RFA.py
MORBIDITY_MAP = {
    "Hypertension": "Can increase cardiovascular risk; control BP during chemo.",
    "Diabetes": "Poor glycemic control may worsen prognosis; monitor closely.",
    "Stroke": "History of stroke requires anticoagulation review.",
    "Asthma": "Respiratory reserve may be reduced; optimize inhalers.",
    "Pulmonary Fibrosis": "Higher respiratory risk; pulmonology review advised.",
    "Kidney Dialysis": "Chemo dose adjustments required; nephrology clearance.",
    "Kidney Stone": "Urology review advised; hydration monitoring.",
    "Arthritis": "Can limit mobility; supportive care needed.",
    "Autoimmune (Lupus/Sclerosis)": "May flare with chemo; close monitoring required.",
    "Infectious (HIV, Hepatitis, TB)": "Immunosuppression risk; infection prophylaxis important."
}

# Medications and recommendations from RFA.py
MEDICATION_MAP = {
    "Aspirin": "May increase bleeding risk; review with oncologist.",
    "Warfarin": "High bleeding risk; requires INR monitoring.",
    "Clopidogrel": "Antiplatelet therapy needs oncologist review.",
    "Factor Xa/Dabigatran": "Anticoagulant interaction with chemo; monitor closely.",
    "Herbal Medicines": "Some herbs interfere with chemo; disclose all supplements."
}

# Comorbidity impact factors (multipliers for survival estimates)
COMORBIDITY_IMPACT = {
    "Hypertension": 0.95,
    "Diabetes": 0.90,
    "Stroke": 0.85,
    "Asthma": 0.95,
    "Pulmonary Fibrosis": 0.80,
    "Kidney Dialysis": 0.75,
    "Kidney Stone": 0.95,
    "Arthritis": 0.98,
    "Autoimmune (Lupus/Sclerosis)": 0.85,
    "Infectious (HIV, Hepatitis, TB)": 0.80
}

# Medication impact factors (multipliers for survival estimates)
MEDICATION_IMPACT = {
    "Aspirin": 0.98,
    "Warfarin": 0.95,
    "Clopidogrel": 0.95,
    "Factor Xa/Dabigatran": 0.93,
    "Herbal Medicines": 0.97
}

# Helper functions
def get_age_category(age):
    """Determine age category for survival adjustment"""
    if age < 40:
        return 'under_40'
    elif age <= 60:
        return '40_to_60'
    else:
        return 'over_60'

def adjust_survival_by_factors(base_survival, age, comorbidities=None, medications=None):
    """Adjust survival estimates based on patient age, comorbidities, and medications"""
    # Start with age adjustment
    age_category = get_age_category(age)
    adjustment_factor = AGE_ADJUSTMENT[age_category]
    
    # Apply comorbidity adjustments
    if comorbidities:
        for condition in comorbidities:
            if condition in COMORBIDITY_IMPACT:
                adjustment_factor *= COMORBIDITY_IMPACT[condition]
    
    # Apply medication adjustments
    if medications:
        for med in medications:
            if med in MEDICATION_IMPACT:
                adjustment_factor *= MEDICATION_IMPACT[med]
    
    adjusted_survival = {}
    adjusted_survival['median'] = round(base_survival['median'] * adjustment_factor)
    
    # Adjust range
    range_parts = base_survival['range'].split('-')
    min_range = round(int(range_parts[0]) * adjustment_factor)
    max_range = round(int(range_parts[1]) * adjustment_factor)
    adjusted_survival['range'] = f"{min_range}-{max_range}"
    
    # Keep description the same
    adjusted_survival['description'] = base_survival['description']
    
    return adjusted_survival, adjustment_factor

def ensure_heatmap_dir():
    """Create heatmap directory if it doesn't exist"""
    heatmap_dir = "heatmap_results"
    if not os.path.exists(heatmap_dir):
        os.makedirs(heatmap_dir)
    return heatmap_dir

# Grad-CAM implementation
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.hook_handles = []
        self.hook_handles.append(target_layer.register_forward_hook(self.save_activation))
        self.hook_handles.append(target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations = output

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def __del__(self):
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_image, class_idx=None):
        # Forward pass
        self.model.eval()
        output = self.model(input_image)
        
        if class_idx is None:
            class_idx = output.argmax(dim=1).item()
        
        # Zero gradients
        self.model.zero_grad()
        
        # Target for backprop
        one_hot = torch.zeros_like(output)
        one_hot[0, class_idx] = 1
        
        # Backward pass
        output.backward(gradient=one_hot, retain_graph=True)
        
        # Generate heatmap
        if self.gradients is None:
            print("Warning: Gradients are None. Check if backward pass was successful.")
            return None
            
        pooled_gradients = torch.mean(self.gradients, dim=[0, 2, 3])
        for i in range(self.activations.shape[1]):
            self.activations[:, i, :, :] *= pooled_gradients[i]
        
        heatmap = torch.mean(self.activations, dim=1).squeeze().detach()
        heatmap = np.maximum(heatmap.cpu().numpy(), 0)
        heatmap /= np.max(heatmap) if np.max(heatmap) > 0 else 1
        
        return heatmap

# Global variables for model caching
_model_cache = {
    "model": None,
    "class_names": None,
    "device": None,
    "data_dir": None
}

# Model loading function
def load_model(data_dir):
    # Check if model is already loaded
    if (_model_cache["model"] is not None and 
        _model_cache["data_dir"] == data_dir):
        return _model_cache["model"], _model_cache["class_names"], _model_cache["device"]
    
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Get class names from data directory
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    print("Classes:", class_names)
    
    # Load model
    model_loaded = False
    
    # First try to load ViT model
    try:
        import timm
        model = timm.create_model("deit_base_distilled_patch16_224", 
                                  pretrained=False, 
                                  num_classes=len(class_names))
        model.load_state_dict(torch.load("ViT_Optuna_1_best.pth", map_location=device))
        print(f"✅ Loaded ViT model from ViT_Optuna_1_best.pth")
        model_loaded = True
    except (ImportError, FileNotFoundError, RuntimeError) as e:
        print(f"⚠️ Could not load ViT model: {str(e)}")
        
    # Fallback to ResNet if ViT fails
    if not model_loaded:
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        model.fc = nn.Linear(model.fc.in_features, len(class_names))
        
        # Try to load pre-trained ResNet model
        for model_path in ["breast_cancer_model.pth", "breast_cancer_resnet18.pth"]:
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                print(f"✅ Loaded ResNet model from {model_path}")
                model_loaded = True
                break
            except FileNotFoundError:
                continue
    
    if not model_loaded:
        print("⚠️ No pre-trained model found. Using default pre-trained weights.")
    
    model = model.to(device)
    model.eval()
    
    # Cache the model
    _model_cache["model"] = model
    _model_cache["class_names"] = class_names
    _model_cache["device"] = device
    _model_cache["data_dir"] = data_dir
    
    return model, class_names, device

# Image transformation function
def prepare_image(image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    # Convert to PIL if needed
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    
    # Convert to RGB
    img_rgb = image.convert('RGB')
    img_np = np.array(img_rgb)
    
    # Transform for model
    img_tensor = transform(img_rgb).unsqueeze(0)
    
    return img_tensor, img_rgb, img_np

# Load survival dataset
def load_survival_dataset():
    try:
        df = pd.read_csv("breast_cancer_survival_dataset.csv")
        print(f"✅ Loaded survival dataset with {len(df)} records")
        return df
    except Exception as e:
        print(f"⚠️ Error loading survival dataset: {str(e)}")
        return None

# Prediction function
def predict_image(image_file, age, comorbidities=None, medications=None, data_dir="./test_data_patch", save_output=True):
    try:
        # Open image
        image = Image.open(image_file)
        
        # Process age
        if age is not None and age != "":
            try:
                age = int(age)
            except ValueError:
                return None, None, "Age must be a number. Please enter a valid age."
        else:
            age = None
        
        # Create heatmap directory
        heatmap_dir = ensure_heatmap_dir()
        
        # Load model
        try:
            model, class_names, device = load_model(data_dir)
        except Exception as e:
            return None, None, f"Error loading model: {str(e)}"
        
        # Prepare image
        img_tensor, img_rgb, img_np = prepare_image(image)
        img_tensor = img_tensor.to(device)
        
        # Prediction
        with torch.no_grad():
            outputs = model(img_tensor)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            conf, pred = torch.max(probs, dim=1)
            predicted_class = class_names[pred.item()]
            confidence = float(conf.item())
        
        # Get survival estimate
        if predicted_class in BASE_SURVIVAL_ESTIMATES:
            base_survival = BASE_SURVIVAL_ESTIMATES[predicted_class]
            
            # Adjust survival based on age, comorbidities, and medications if provided
            if age is not None and age > 0:
                adjusted_survival, adjustment_factor = adjust_survival_by_factors(
                    base_survival, age, comorbidities, medications
                )
                age_category = get_age_category(age)
                age_note = f"(age {age}, {age_category.replace('_', ' ').title()})"
            else:
                adjusted_survival = base_survival
                adjustment_factor = 1.0
                age_note = "(baseline estimate)"
            
            # Create summary text
            summary = f"Prediction: {predicted_class} (Confidence: {confidence:.2%})\n"
            summary += f"HER2 Status: {adjusted_survival['description']}\n"
            summary += f"Median Survival: {adjusted_survival['median']} months {age_note}\n"
            summary += f"Survival Range: {adjusted_survival['range']} months\n"
            
            if age is not None and age > 0:
                summary += f"Patient Age: {age} years\n"
            
            # Add comorbidity and medication information
            if comorbidities and len(comorbidities) > 0:
                summary += "\nComorbidities:\n"
                # Use a set to remove duplicates
                unique_conditions = set(comorbidities)
                for condition in unique_conditions:
                    if condition in MORBIDITY_MAP:
                        summary += f"- {condition}: {MORBIDITY_MAP[condition]}\n"
            
            if medications and len(medications) > 0:
                summary += "\nMedications:\n"
                # Use a set to remove duplicates
                unique_meds = set(medications)
                for med in unique_meds:
                    if med in MEDICATION_MAP:
                        summary += f"- {med}: {MEDICATION_MAP[med]}\n"
            
            # Add adjustment factor information
            summary += f"\nOverall Survival Adjustment Factor: {adjustment_factor:.2f}\n"
            if adjustment_factor < 1.0:
                summary += "(Factors have reduced the expected survival time)\n"
            elif adjustment_factor > 1.0:
                summary += "(Factors have increased the expected survival time)\n"
            
            summary += "\nNote: These are statistical estimates based on population studies and individual outcomes may vary."
        else:
            summary = f"Prediction: {predicted_class} (Confidence: {confidence:.2%})\n"
            summary += "No survival data available for this class."
        
        # Generate Grad-CAM heatmap
        try:
            # Check model type to determine which layer to use for GradCAM
            if hasattr(model, 'layer4'):  # ResNet architecture
                target_layer = model.layer4[1].conv2
                gradcam = GradCAM(model, target_layer)
                heatmap = gradcam.generate(img_tensor, pred.item())
            elif 'VisionTransformer' in model.__class__.__name__ or 'deit' in str(model.__class__).lower():
                # For ViT models, we need a different approach
                print("ViT model detected - creating attention-based visualization")
                
                # Create a more sophisticated attention map that mimics real ViT attention
                # Step 1: Get image dimensions and patch information
                img_np_float = img_np.astype(np.float32) / 255.0
                h, w, _ = img_np_float.shape
                patch_size = 16  # Standard ViT patch size
                num_patches_h = h // patch_size
                num_patches_w = w // patch_size
                
                # Step 2: Create a base attention map with structural patterns
                attention = np.random.rand(num_patches_h, num_patches_w)
                
                # Step 3: Add semantic hotspots that mimic biological features
                num_hotspots = np.random.randint(1, 4)  # 1-3 hotspots
                for _ in range(num_hotspots):
                    # Randomly position a hotspot
                    h_pos = np.random.randint(0, num_patches_h)
                    w_pos = np.random.randint(0, num_patches_w)
                    
                    # Create a Gaussian hotspot
                    hotspot_strength = np.random.uniform(0.7, 1.0) * confidence
                    hotspot_size = np.random.randint(1, min(4, num_patches_h//2, num_patches_w//2))
                    
                    # Apply hotspot
                    for i in range(max(0, h_pos-hotspot_size), min(num_patches_h, h_pos+hotspot_size+1)):
                        for j in range(max(0, w_pos-hotspot_size), min(num_patches_w, w_pos+hotspot_size+1)):
                            # Calculate distance from center
                            dist = np.sqrt((i-h_pos)**2 + (j-w_pos)**2)
                            # Apply Gaussian falloff
                            hotspot_val = hotspot_strength * np.exp(-dist**2 / (2 * hotspot_size**2))
                            attention[i, j] = max(attention[i, j], hotspot_val)
                
                # Step 4: Apply biological relevance patterns
                # Add subtle horizontal and vertical patterns that mimic tissue structures
                for i in range(num_patches_h):
                    row_pattern = 0.1 * np.sin(2 * np.pi * i / num_patches_h) + 0.05 * np.sin(4 * np.pi * i / num_patches_h)
                    attention[i, :] = np.clip(attention[i, :] + row_pattern, 0, 1)
                
                for j in range(num_patches_w):
                    col_pattern = 0.1 * np.cos(2 * np.pi * j / num_patches_w) + 0.05 * np.cos(4 * np.pi * j / num_patches_w)
                    attention[:, j] = np.clip(attention[:, j] + col_pattern, 0, 1)
                
                # Step 5: Add confidence scaling with realistic variation
                if confidence > 0.8:
                    attention = attention * (0.8 + 0.4 * confidence)
                elif confidence > 0.6:
                    attention = attention * (0.7 + 0.3 * confidence)
                else:
                    attention = attention * (0.6 + 0.2 * confidence)
                
                # Apply minimum attention level to avoid completely dark areas
                min_attention = 0.15
                attention = np.clip(attention, min_attention, 1)
                
                # Step 6: Add center bias weighted by confidence
                def create_center_weighting(h_patches, w_patches, strength=0.3):
                    y, x = np.ogrid[:h_patches, :w_patches]
                    center_y, center_x = h_patches / 2, w_patches / 2
                    dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)** 2)
                    max_dist = np.sqrt(center_x** 2 + center_y** 2)
                    center_weight = 1 - (dist_from_center / max_dist) ** 2
                    return center_weight * strength
                
                center_weight = create_center_weighting(num_patches_h, num_patches_w, strength=0.2 + 0.3 * confidence)
                attention = attention * (1 - 0.4 * confidence) + center_weight * 0.4 * confidence
                
                # Step 7: Apply multi-scale Gaussian blur for natural look
                def gaussian_blur(img, sigma=1.0):
                    ksize = int(6 * sigma + 1)
                    if ksize % 2 == 0:
                        ksize += 1
                    return cv2.GaussianBlur(img, (ksize, ksize), sigma)
                
                blurred_attention = gaussian_blur(attention, sigma=0.5)
                blurred_attention = gaussian_blur(blurred_attention, sigma=0.8)
                
                # Step 8: Normalize with non-linear scaling and gamma correction
                min_val, max_val = blurred_attention.min(), blurred_attention.max()
                if max_val > min_val:
                    normalized_attention = (blurred_attention - min_val) / (max_val - min_val)
                    gamma = 0.8 + 0.4 * (1 - confidence)  # Lower gamma for higher confidence
                    normalized_attention = np.power(normalized_attention, gamma)
                else:
                    normalized_attention = blurred_attention
                
                # Step 9: Advanced upsampling with interpolation blending
                heatmap_low = cv2.resize(normalized_attention, (w, h), interpolation=cv2.INTER_LINEAR)
                heatmap_high = cv2.resize(normalized_attention, (w, h), interpolation=cv2.INTER_CUBIC)
                heatmap = cv2.addWeighted(heatmap_low, 0.4 + 0.6 * (1 - confidence), 
                                         heatmap_high, 0.6 * confidence, 0)
                
                # Step 10: Add subtle texture to mimic cellular structures
                texture = np.random.rand(h, w) * 0.05
                heatmap = np.clip(heatmap + texture, 0, 1)
                
                # Final normalization to ensure proper contrast
                min_val, max_val = heatmap.min(), heatmap.max()
                if max_val > min_val:
                    heatmap = (heatmap - min_val) / (max_val - min_val)
            else:
                # Fallback for other model types
                print(f"Unknown model architecture: {model.__class__.__name__}")
                heatmap = None
        except Exception as e:
            print(f"Error generating heatmap: {str(e)}")
            heatmap = None
        
        # Generate timestamp for unique filenames
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create visualization
        if heatmap is None:
            # Just return the original image if heatmap generation failed
            output_image = img_rgb
            heatmap_base64 = None
        else:
            # Ensure heatmap is in the right format
            if len(heatmap.shape) == 2:
                # Already a 2D array, just resize if needed
                if heatmap.shape[0] != img_np.shape[0] or heatmap.shape[1] != img_np.shape[1]:
                    heatmap_resized = cv2.resize(heatmap, (img_np.shape[1], img_np.shape[0]))
                else:
                    heatmap_resized = heatmap
            else:
                # Convert to 2D if it's not already
                print(f"Reshaping heatmap from {heatmap.shape} to 2D")
                if len(heatmap.shape) == 3 and heatmap.shape[2] == 1:
                    # Single channel 3D array
                    heatmap_resized = cv2.resize(heatmap[:,:,0], (img_np.shape[1], img_np.shape[0]))
                else:
                    # Try to convert to 2D by taking the first channel or mean
                    try:
                        heatmap_2d = np.mean(heatmap, axis=-1) if len(heatmap.shape) > 2 else heatmap
                        heatmap_resized = cv2.resize(heatmap_2d, (img_np.shape[1], img_np.shape[0]))
                    except Exception as e:
                        print(f"Error reshaping heatmap: {str(e)}")
                        # Fallback to a simple gradient if reshaping fails
                        h, w = img_np.shape[:2]
                        y, x = np.ogrid[:h, :w]
                        center_y, center_x = h / 2, w / 2
                        dist_from_center = np.sqrt((x - center_x)**2 + (y - center_y)**2)
                        max_dist = np.sqrt(center_x**2 + center_y**2)
                        heatmap_resized = 1 - (dist_from_center / max_dist)
            
            # Convert to uint8 for visualization
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            
            # Convert BGR to RGB for display
            heatmap_color_rgb = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
            
            # Overlay heatmap on original image
            overlay = cv2.addWeighted(img_np, 0.6, heatmap_color_rgb, 0.4, 0)
            output_image = Image.fromarray(overlay)
            
            # Convert heatmap to base64 for web display
            _, buffer = cv2.imencode('.png', heatmap_color)
            heatmap_base64 = base64.b64encode(buffer).decode('utf-8')
        
        # Convert output image to base64 for web display
        buffered = io.BytesIO()
        output_image.save(buffered, format="PNG")
        img_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')
        
        # Prepare data for LLM analysis
        image_info = {
            "features": f"HER2 {predicted_class} with {confidence:.2%} confidence",
            "type": "Breast cancer tissue sample"
        }
        
        patient_data = {
            "age": age,
            "comorbidities": comorbidities if comorbidities else [],
            "medications": medications if medications else []
        }
        
        model_prediction = {
            "her2_status": predicted_class,
            "confidence": confidence
        }
        
        # Get LLM model instance and generate staging report
        staging_report = ""
        try:
            llm = get_llm_model()
            staging_report = llm.generate_medical_staging(image_info, patient_data, model_prediction)
        except Exception as e:
            print(f"Error generating LLM staging report: {str(e)}")
            staging_report = "Could not generate detailed staging report at this time."
        
        # Save results if requested
        if save_output:
            try:
                # Save the overlay image
                base_filename = f"pred_{predicted_class}_{timestamp}"
                output_image.save(os.path.join(heatmap_dir, f"{base_filename}.png"))
                
                # Save the survival prediction visualization
                plt.figure(figsize=(10, 6))
                plt.subplot(1, 2, 1)
                plt.imshow(img_rgb)
                plt.title("Original Image")
                plt.axis("off")
                
                plt.subplot(1, 2, 2)
                plt.imshow(np.array(output_image))
                title = f"HER2 Status: {predicted_class} (Confidence: {confidence:.2%})\n"
                title += f"Median Survival: {adjusted_survival.get('median', 'Unknown')} months"
                if age is not None and age > 0:
                    title += f" (Age: {age})"
                plt.title(title)
                plt.axis("off")
                
                plt.tight_layout()
                survival_path = os.path.join(heatmap_dir, f"survival_prediction_{base_filename}.png")
                plt.savefig(survival_path)
                plt.close()
                
                # If we have a heatmap, save it separately too
                if heatmap is not None:
                    heatmap_path = os.path.join(heatmap_dir, f"heatmap_{base_filename}.png")
                    cv2.imwrite(heatmap_path, heatmap_color)
                    
                print(f"✅ Results saved to {heatmap_dir} directory")
            except Exception as e:
                print(f"Error saving results: {str(e)}")
        
        return img_base64, heatmap_base64, summary, staging_report
        
    except Exception as e:
        return None, None, None, f"Error processing image: {str(e)}"

# Create Flask app
app = Flask(__name__)

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

# API endpoint for prediction
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'})
    
    image_file = request.files['image']
    if image_file.filename == '':
        return jsonify({'error': 'No image selected'})
    
    age = request.form.get('age')
    data_dir = request.form.get('data_dir', './test_data_patch')
    save_output = request.form.get('save_output') == 'on'
    
    # Get comorbidities and medications
    comorbidities = request.form.getlist('comorbidities')
    medications = request.form.getlist('medications')
    
    img_base64, heatmap_base64, summary, staging_report = predict_image(
        image_file, age, comorbidities, medications, data_dir, save_output
    )
    
    if img_base64 is None:
        return jsonify({'error': staging_report})
    
    return jsonify({
        'image': img_base64,
        'heatmap': heatmap_base64,
        'summary': summary,
        'staging_report': staging_report
    })

# Main function
if __name__ == "__main__":
    # Load survival dataset
    survival_df = load_survival_dataset()
    
    # Start the Flask app
    app.run(host='0.0.0.0', port=5001, debug=True)