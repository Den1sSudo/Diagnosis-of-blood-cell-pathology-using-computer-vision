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

# To run this project you need
1. At first you need to train a model train_module.py
2. For training, you can use your already prepared data, or use other people's data from open sources.
3. When you model is ready launch analyze.py
