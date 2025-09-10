# Prediction-of-Ovarian-Cancer-Using-Machine-Learning-Techniques-

This project focuses on the **early detection and classification of ovarian cancer** using **Machine Learning (ML) and Convolutional Neural Networks (CNNs)**. By analyzing **histopathological images**, the system identifies cancerous regions and predicts stages, aiming to improve early diagnosis and treatment outcomes.

---

#  Features
- Image preprocessing: noise removal, thresholding, edge detection, image sharpening  
- Feature extraction: texture, shape descriptors, Sobel gradient  
- CNN-based classification of histopathological images  
- Detection of **benign vs malignant** cases  
- Cell counting: total, damaged, and overlapping cells  
- Model evaluation with **accuracy, loss graphs, confusion matrix, precision & F1-score**  
- Simple interface for image input and output visualization  

---

# Project Structure
-data/ Dataset (histopathological images)
-src/ # Source code (Python scripts for preprocessing, training, testing)
-models/ # Trained CNN models
-results/ # Graphs, confusion matrix, performance metrics
-requirements.txt # Python dependencies


# Requirements
- Python **3.7+**
- Libraries:
  - `numpy`
  - `opencv-python`
  - `matplotlib`
  - `tensorflow` / `keras`
  - `scikit-learn`
- (Optional) **MATLAB** (for initial prototyping and preprocessing experiments)
