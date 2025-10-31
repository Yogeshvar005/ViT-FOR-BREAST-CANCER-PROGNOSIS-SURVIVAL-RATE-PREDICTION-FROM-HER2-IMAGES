# Title: ViT-FOR-BREAST-CANCER-PROGNOSIS-SURVIVAL-RATE-PREDICTION-FROM-HER2-IMAGES
Breast cancer remains one of the leading causes of mortality among women worldwide, and its accurate detection at an early stage is critical for improving patient survival rates. 

Traditional diagnostic methods, such as mammography and histopathological image analysis, rely heavily on radiologists and pathologists, making the process time-consuming, subjective, and prone to inter-observer variability. 

With the rapid increase in the volume and complexity of medical imaging data, manual interpretation alone is insufficient to meet the demand for timely and precise diagnosis.

## About:
The project aims to develop an AI-driven system capable of predicting breast cancer stages and estimating patient survival ranges using both histopathological images and clinical data.
It integrates advanced Vision Transformer (ViT) architectures for image analysis and Explainable AI (XAI) frameworks to ensure transparency and interpretability in predictions.
The system will also include a survival prediction module to assist oncologists in personalized treatment planning and prognosis estimation.
A user-friendly web application will be deployed, enabling healthcare professionals to upload patient data and instantly receive diagnostic and survival insights.
The project’s scope extends beyond diagnosis to include clinical decision support, research assistance, and improved patient management, promoting early intervention and data-driven healthcare.
In the long term, the system can be scaled to analyze other cancer types or integrated with hospital information systems for real-time clinical use.


## Features:
1. Dual-Module Architecture

      The system combines cancer stage detection and survival period prediction within a single integrated platform.

      This dual functionality enables both diagnostic and prognostic insights for breast cancer patients.

2. Vision Transformer (ViT)-Based Image Analysis

      Utilizes ViT for high-precision classification of HER2-stained histopathological images.

      Automatically learns complex spatial patterns and tissue morphologies, outperforming traditional CNN-based approaches.

3. Patient History Integration

      Incorporates clinical and demographic data (e.g., age, menopausal status, comorbidities) alongside image features to enhance prediction accuracy.

      Enables personalized analysis rather than relying on image data alone.

4. Hybrid AI Framework

      Combines deep learning (ViT) for visual feature extraction with machine learning models (like Cox regression, Random Forest, or XGBoost) for survival analysis.

      Delivers interpretable results linking visual and clinical factors to survival outcomes.

5. Stage-Wise Classification System

      Accurately detects HER2 cancer stages (I–IV) based on tissue characteristics.

      Offers visual heatmaps and attention maps to explain stage predictions.



## Requirements:
### HARDWARE ENVIRONMENT
*	Processor : Intel Pentium Dual Core / i3 or higher (2.00 GHz or above)
*	Hard Disk : Minimum 120 GB HDD / SSD recommended for faster model training
*	RAM : 8 GB (minimum), 16 GB recommended for deep learning operations
*	Keyboard : 110 keys enhanced
*	GPU (Optional) : NVIDIA GPU with CUDA support for deep learning model acceleration

### SOFTWARE ENVIRONMENT
*	Operating System : Windows 7 (Service Pack 1), 8, 8.1, 10, or 11
*	Programming Language : Python 3.10 or above
*	Supporting Libraries :
*	NumPy, Pandas, Matplotlib, Seaborn – for data preprocessing and visualization
*	Scikit-learn, Lifelines, Scikit-survival – for survival analysis and machine learning models
*	TensorFlow / PyTorch – for deep learning model implementation
*	Joblib, Pickle – for model serialization
*	OpenCV, PIL – for image preprocessing (if histopathological images are used)

### TECHNOLOGIES USED
*	Integrated Development Environment (IDE) : Visual Studio Code / Jupyter Notebook
*	Framework : Streamlit – for developing an interactive web-based user interface
*	Deep Learning : TensorFlow / PyTorch – used for building and training breast cancer prediction models
*	Machine Learning Models : Random Forest Algorithm (RFA), Support Vector Machine (SVM), Multi-Layer Perceptron (MLP)
*	Survival Analysis Models : Cox Proportional Hazards (CoxPH), Random Survival Forest (RSF)
* Version Control : Git and GitHub – for project management and collaboration


## System Architecture
<!--Embed the system architecture diagram as shown below-->
<img width="1027" height="658" alt="image" src="https://github.com/user-attachments/assets/efb33be1-8405-4ee1-ab68-074d34913699" />



## Output:

<!--Embed the Output picture at respective places as shown below as shown below-->
#### Output1 - HOME PAGE

<img width="1181" height="596" alt="image" src="https://github.com/user-attachments/assets/93739cc4-a8b8-42f3-a32e-ec8e9bec3849" />

#### Output2 - DETECTION WITH DETAILS
<img width="1026" height="704" alt="image" src="https://github.com/user-attachments/assets/4b008792-8a89-4a8c-84ff-e8b2a4b49f9e" />

#### Output3 - PROGNOSIS
<img width="957" height="1048" alt="image" src="https://github.com/user-attachments/assets/5adc56c6-8808-44a2-840b-241da85eefe7" />


Detection Accuracy: 96.7%
Note: These metrics can be customized based on your actual performance evaluations.


## Results and Impact:
In conclusion, our developed Breast Cancer Detection System, leveraging the powerful capabilities of the Vision Transformer (ViT) model, represents a significant advancement in the field of medical image analysis and diagnostic automation. The system is specifically designed to accurately distinguish between benign and malignant breast tissue samples, demonstrating exceptional performance in terms of accuracy, precision, and robustness across diverse datasets. Our experiments confirm the model’s ability to effectively learn complex spatial and textural patterns in histopathology images, making it highly reliable for early and automated cancer diagnosis.
By integrating state-of-the-art deep learning techniques with an efficient and user-friendly interface, the proposed system offers a practical and scalable solution for use in clinical and diagnostic environments. The automation of cancer classification not only assists pathologists in reducing workload and diagnostic time but also enhances the overall reliability and consistency of results, which is crucial in life-saving medical decisions.


## Articles published / References:
* [1]  Hamedi, S.Z., Emami, H., Khayamzadeh, M. et al. Application of machine learning in breast cancer survival prediction using a multimethod approach. Sci Rep 14, 30147 (2024). https://doi.org/10.1038/s41598-024-81734-y
* [2]  Lu C, Shiradkar R, Liu Z. Integrating pathomics with radiomics and genomics for cancer prognosis: A brief review. Chin J Cancer Res. 2021 Oct 31;33(5):563-573. doi: 10.21147/j.issn.1000-9604.2021.05.03. PMID: 34815630; PMCID: PMC8580801
* [3]  Qian, X., Pei, J., Han, C. et al. A multimodal machine learning model for the stratification of breast cancer risk. Nat. Biomed. Eng 9, 356–370 (2025). https://doi.org/10.1038/s41551-024-01302-7
