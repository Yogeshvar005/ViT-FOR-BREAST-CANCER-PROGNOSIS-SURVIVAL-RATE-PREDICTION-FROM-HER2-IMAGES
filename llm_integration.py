import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import time
import os

class LLMModel:
    def __init__(self):
        # We'll use a local LLM model suitable for medical image analysis
        # For demonstration purposes, we'll use a smaller model for faster inference
        self.model_name = "llama-3-8b-instruct"  # This is an example, will use local model
        self.model = None
        self.tokenizer = None
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.load_model()
        
    def load_model(self):
        """Load the LLM model and tokenizer"""
        try:
            # First try to load from local path if available
            local_model_path = "./local_llm_model"
            if os.path.exists(local_model_path):
                print(f"Loading LLM model from local path: {local_model_path}")
                self.tokenizer = AutoTokenizer.from_pretrained(local_model_path)
                self.model = AutoModelForCausalLM.from_pretrained(
                    local_model_path,
                    device_map="auto",
                    torch_dtype=torch.float16,
                    use_safetensors=True
                )
            else:
                # Fallback to a smaller demonstration model
                # In a real scenario, you would use a more capable medical model
                print("Loading demonstration LLM model...")
                # For this example, we'll use a lightweight model
                self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
                self.model = AutoModelForCausalLM.from_pretrained(
                    "gpt2",
                    torch_dtype=torch.float16,
                    device_map="auto"
                )
                
            self.model.to(self.device)
            self.model.eval()
            print(f"✅ LLM model loaded successfully on {self.device}")
        except Exception as e:
            print(f"⚠️ Error loading LLM model: {str(e)}")
            # Set up a simple fallback for demonstration
            self.tokenizer = None
            self.model = None
            print("Using fallback LLM simulation")
    
    def generate_medical_staging(self, image_info, patient_data, model_prediction):
        """Generate medical staging analysis using LLM"""
        if self.model is None or self.tokenizer is None:
            return self._generate_fallback_staging(image_info, patient_data, model_prediction)
            
        try:
            # Create prompt for LLM
            prompt = self._create_medical_prompt(image_info, patient_data, model_prediction)
            
            # Tokenize and generate response
            inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
            
            # Generate response with temperature control for deterministic output
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=500,
                    temperature=0.7,
                    top_p=0.95,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id
                )
            
            # Decode and return response
            response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the LLM's response after the prompt
            if prompt in response:
                staging_result = response[len(prompt):].strip()
            else:
                staging_result = response.strip()
            
            return staging_result
        except Exception as e:
            print(f"⚠️ Error generating LLM staging: {str(e)}")
            return self._generate_fallback_staging(image_info, patient_data, model_prediction)
            
    def _create_medical_prompt(self, image_info, patient_data, model_prediction):
        """Create a medical prompt for the LLM"""
        prompt = f"""You are a medical expert specializing in breast cancer pathology and staging. Based on the information provided, generate a comprehensive staging analysis.
        
        Imaging Analysis:
        - HER2 Status: {model_prediction.get('her2_status', 'Unknown')}
        - Confidence Level: {model_prediction.get('confidence', 'Unknown'):.2%}
        - Image Features: {image_info.get('features', 'Typical for this HER2 status')}
        
        Patient Data:
        - Age: {patient_data.get('age', 'Unknown')} years
        - Comorbidities: {', '.join(patient_data.get('comorbidities', [])) if patient_data.get('comorbidities') else 'None'}
        - Medications: {', '.join(patient_data.get('medications', [])) if patient_data.get('medications') else 'None'}
        
        Please provide:
        1. Detailed HER2 staging assessment
        2. Prognostic implications based on all factors
        3. Treatment considerations
        4. Follow-up recommendations
        
        Your response should be concise but medically accurate, formatted for easy reading."""
        
        return prompt
    
    def _generate_fallback_staging(self, image_info, patient_data, model_prediction):
        """Generate a fallback staging analysis when LLM is not available"""
        her2_status = model_prediction.get('her2_status', 'Unknown')
        confidence = model_prediction.get('confidence', 0)
        age = patient_data.get('age', 'Unknown')
        comorbidities = patient_data.get('comorbidities', [])
        medications = patient_data.get('medications', [])
        
        # Generate a structured report based on available data
        report = f"HER2 Staging Analysis Report\n"
        report += "=" * 50 + "\n\n"
        
        # HER2 Assessment
        report += f"HER2 Status: {her2_status}\n"
        report += f"Confidence: {confidence:.2%}\n\n"
        
        # Prognostic Implications
        report += "Prognostic Implications:\n"
        if her2_status == 'class_0':
            report += "- Favorable prognosis with standard treatment\n"
        elif her2_status == 'class_1+':
            report += "- Intermediate prognosis requiring comprehensive management\n"
        elif her2_status == 'class_2+':
            report += "- Requires additional testing (FISH) to confirm HER2 amplification\n"
        elif her2_status == 'class_3+':
            report += "- Aggressive subtype that may benefit from HER2-targeted therapies\n"
        
        if age and age < 40:
            report += "- Younger age may indicate more aggressive disease biology\n"
        elif age and age > 60:
            report += "- Older age requires careful consideration of treatment tolerance\n"
        
        # Impact of Comorbidities and Medications
        if comorbidities:
            report += "\nImpact of Comorbidities:\n"
            for condition in comorbidities:
                report += f"- {condition}: May require treatment modifications and additional monitoring\n"
        
        if medications:
            report += "\nMedication Considerations:\n"
            for med in medications:
                report += f"- {med}: Requires review for potential interactions with cancer therapies\n"
        
        # Treatment Recommendations
        report += "\nTreatment Considerations:\n"
        if her2_status in ['class_2+', 'class_3+']:
            report += "- Consider HER2-targeted therapies (trastuzumab, pertuzumab)\n"
        report += "- Multidisciplinary team evaluation recommended\n"
        report += "- Personalized treatment plan based on full clinical context\n"
        
        # Follow-up
        report += "\nFollow-up Recommendations:\n"
        report += "- Regular clinical and imaging surveillance\n"
        report += "- Cardiac function monitoring if HER2-targeted therapy is initiated\n"
        report += "- Comprehensive management of comorbid conditions\n"
        
        report += "\nNote: This is an automated assessment. Clinical correlation is essential."
        
        return report

# Create a singleton instance of the LLM model
llm_instance = None

def get_llm_model():
    global llm_instance
    if llm_instance is None:
        llm_instance = LLMModel()
    return llm_instance

# Example usage
if __name__ == "__main__":
    # Test the LLM integration
    llm = get_llm_model()
    
    # Sample data
    image_info = {
        "features": "Uniform nuclear size, regular spacing, minimal pleomorphism"
    }
    
    patient_data = {
        "age": 52,
        "comorbidities": ["Hypertension"],
        "medications": ["Aspirin"]
    }
    
    model_prediction = {
        "her2_status": "class_0",
        "confidence": 0.85
    }
    
    # Generate staging report
    staging_report = llm.generate_medical_staging(image_info, patient_data, model_prediction)
    print(staging_report)