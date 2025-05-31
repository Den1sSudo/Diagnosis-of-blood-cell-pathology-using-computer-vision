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
![Image](https://github.com/user-attachments/assets/09cfe912-2c05-47ed-b6c9-7fe1bfc73913)
## Information about diagnosys 
![img_7.png](img_7.png)

# To run this project you need
1. At first you need to train a model train_module.py
2. For training, you can use your already prepared data, or use other people's data from open sources. For that project im use: https://aslan.md/blood-cell-detection-dataset/
3. When you model is ready launch analyze.py
