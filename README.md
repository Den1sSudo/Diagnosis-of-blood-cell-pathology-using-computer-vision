# The purpose
Manual counting and evaluation of blood cells under a microscope takes from 30 minutes to several hours per sample. In the context of mass research (for example, during pandemics) it takes a lot of effort. Staff fatigue leads to missing anomalies.  In regions with a shortage of qualified hematologists, diagnoses are often delayed, which worsens the prognosis of patients.
# Project Modules
The project consists of two modules: 1. Training module (train_module) and 2. Analysis and Interface module (analyze)
1.Reads marked-up images
- Extracts signs from cells
- Trains 3 classifiers
- Saves the trained models in models/
2. Loads models
- Calls extract_features() for new images
- Classifies each cell
- Displays the results to the user
## Extract features
![Image](https://github.com/user-attachments/assets/0389d583-a390-4576-a2d2-62c58bbe6c2b)

## Cells 
Signs of cells:

- area, perimeter — the area and perimeter
- roundness — the degree of roundness
- brightness — brightness in gray
- color_type — hue classification (HSV)

## The color by the HSV mask:
![Image](https://github.com/user-attachments/assets/c4e10497-bf9d-4a46-9c70-158b83ba2943)

## All operations are performed using OpenCV:

- conversion to gray and HSV format
- search for cell contours
- creation of masks for cell selection
- calculation of geometric and color features

## Model training and preservation
Model: Random Forest Classifier
![Image](https://github.com/user-attachments/assets/724c2895-13cc-46b9-b132-078b0061169c)

- n_estimators=100 → 100 decision trees
- 3 classifiers are created:
- clf_cell: cell type (leukocyte/erythrocyte)
- clf_ery: for erythrocytes
- clf_ley: for leukocytes

## Saving models
![Image](https://github.com/user-attachments/assets/741e1863-7a8b-41ae-9d9d-a5b4740f97ad)

## User interface
![Image](https://github.com/user-attachments/assets/d85f6b35-0549-4452-add1-50f7983a67b5)

## Results
![Image](https://github.com/user-attachments/assets/7a6aff34-7464-4e0f-a887-af6cf54c7025)
## Information about diagnosys 
![Image](https://github.com/user-attachments/assets/ba9fc422-e2d0-4b51-85b2-4378bed5f9de)

# To run this project you need
1. For install all librares you can use file with requirements.txt
2. At first you need to train a model train_module.py
3. For training, you can use your already prepared data, or use other people's data from open sources. For that project im use: https://aslan.md/blood-cell-detection-dataset/
4. When you model is ready launch analyze.py
