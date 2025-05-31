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
![img.png](img.png)

## Cells 
Signs of cells:

- area, perimeter — the area and perimeter
- roundness — the degree of roundness
- brightness — brightness in gray
- color_type — hue classification (HSV)

## The color is determined by the HSV mask:
![img_2.png](img_2.png)

## All operations are performed using OpenCV:

- conversion to gray and HSV format
- search for cell contours
- creation of masks for cell selection
- calculation of geometric and color features

## Model training and preservation
Model: Random Forest Classifier
![img_3.png](img_3.png)

- n_estimators=100 → 100 decision trees
- 3 classifiers are created:
- clf_cell: cell type (leukocyte/erythrocyte)
- clf_ery: for erythrocytes
- clf_ley: for leukocytes

## Saving models
![img_4.png](img_4.png)

## User interface
![img_5.png](img_5.png)

## Results
![img_6.png](img_6.png)

## Information about diagnosys 
![img_7.png](img_7.png)

# To run this project you need
1. At first you need to train a model train_module.py
2. For training, you can use your already prepared data, or use other people's data from open sources. For that project im use: https://aslan.md/blood-cell-detection-dataset/
3. When you model is ready launch analyze.py
