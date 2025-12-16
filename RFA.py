import gradio as gr

# Morbidities and recommendations
morbidity_map = {
    "Hypertension": "Can increase cardiovascular risk; control BP during chemo.",
    "Diabetes": "Poor glycemic control may worsen prognosis; monitor closely.",
    "Stroke": "History of stroke requires anticoagulation review.",
    "Asthma": "Respiratory reserve may be reduced; optimize inhalers.",
    "Pulmonary Fibrosis": "Higher respiratory risk; pulmonology review advised.",
    "Kidney Dialysis": "Chemo dose adjustments required; nephrology clearance.",
    "Kidney Stone": "Urology review advised; hydration monitoring.",
    "Arthritis": "Can limit mobility; supportive care needed.",
    "Autoimmune (Lupus/Sclerosis)": "May flare with chemo; close monitoring required.",
    "Infectious (HIV, Hepatitis, TB)": "Immunosuppression risk; infection prophylaxis important.",
    "UTI/Dental Infection": "Infection must be treated before chemo starts."
}

# Medications and recommendations
medication_map = {
    "Aspirin": "May increase bleeding risk; review with oncologist.",
    "Warfarin": "High bleeding risk; requires INR monitoring.",
    "Clopidogrel": "Antiplatelet therapy needs oncologist review.",
    "Factor Xa/Dabigatran": "Anticoagulant interaction with chemo; monitor closely.",
    "Herbal Medicines": "Some herbs interfere with chemo; disclose all supplements."
}

def generate_recommendations(selected_morbidities, selected_meds):
    output = "### You selected:\n"
    
    if selected_morbidities:
        output += "\n**Morbidities:**\n"
        for m in selected_morbidities:
            output += f"- **{m}** → {morbidity_map[m]}\n"
    
    if selected_meds:
        output += "\n**Medications:**\n"
        for med in selected_meds:
            output += f"- **{med}** → {medication_map[med]}\n"
    
    if selected_morbidities or selected_meds:
        output += "\n⚠️ These factors can alter your life expectancy with breast cancer. Please discuss with your oncologist."
    else:
        output += "\n✅ No comorbidities/medications selected. Continue routine check-ups."
    
    return output

# Gradio interface
with gr.Blocks() as demo:
    gr.Markdown("# 🩺 Breast Cancer Comorbidity & Medication Checklist")
    gr.Markdown("Tick any conditions or medicines that apply to you:")

    morbidity_input = gr.CheckboxGroup(choices=list(morbidity_map.keys()), label="Select Morbidities")
    med_input = gr.CheckboxGroup(choices=list(medication_map.keys()), label="Select Medications")

    submit_btn = gr.Button("Submit")
    output_box = gr.Markdown()

    submit_btn.click(fn=generate_recommendations, inputs=[morbidity_input, med_input], outputs=output_box)

demo.launch()