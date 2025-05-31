import os
import cv2
import numpy as np
import pandas as pd
import warnings
import joblib
from sklearn.ensemble import RandomForestClassifier

warnings.filterwarnings('ignore', category=UserWarning)

def extract_features(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return None
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, binary = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = cv2.bitwise_not(binary)
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    features_list = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 50:
            continue
        perimeter = cv2.arcLength(cnt, True)
        roundness = (4 * np.pi * area) / (perimeter ** 2 + 1e-5)
        mask = np.zeros(gray.shape, dtype=np.uint8)
        cv2.drawContours(mask, [cnt], -1, 255, -1)
        mean_intensity = cv2.mean(gray, mask=mask)[0]
        mean_h = cv2.mean(img_hsv[:, :, 0], mask=mask)[0]
        mean_s = cv2.mean(img_hsv[:, :, 1], mask=mask)[0]
        color_type = 'leukocyte' if 100 <= mean_h <= 140 and mean_s > 50 else 'erythrocyte'
        features_list.append(([area, perimeter, roundness, mean_intensity], cnt, color_type))
    return features_list

def load_dataset(data_dir='data_cells'):
    data = []
    for cell_type, subtypes in {
        'erythrocyte': ['microcytosis', 'macrocytosis', 'norm'],
        'leukocyte': ['leukocytosis', 'leukopenia', 'norm']
    }.items():
        for subtype in subtypes:
            folder = os.path.join(data_dir, cell_type, subtype)
            if not os.path.exists(folder):
                continue
            for fname in os.listdir(folder):
                if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    continue
                img_path = os.path.join(folder, fname)
                features = extract_features(img_path)
                if features is None:
                    continue
                for feat, _, _ in features:
                    data.append({
                        'cell_type': cell_type,
                        'subtype': subtype,
                        'area': feat[0],
                        'perimeter': feat[1],
                        'roundness': feat[2],
                        'brightness': feat[3]
                    })
    return pd.DataFrame(data)

def train_models(df):
    clf_cell = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_cell.fit(df[['area', 'perimeter', 'roundness', 'brightness']], df['cell_type'])

    clf_ery = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_ery.fit(df[df['cell_type'] == 'erythrocyte'][['area', 'perimeter', 'roundness', 'brightness']],
                df[df['cell_type'] == 'erythrocyte']['subtype'])

    clf_ley = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_ley.fit(df[df['cell_type'] == 'leukocyte'][['area', 'perimeter', 'roundness', 'brightness']],
                df[df['cell_type'] == 'leukocyte']['subtype'])

    return clf_cell, clf_ery, clf_ley

if __name__ == '__main__':
    df = load_dataset()
    if df.empty:
        print('Данные не найдены.')
    else:
        clf_cell, clf_ery, clf_ley = train_models(df)
        os.makedirs('models', exist_ok=True)
        joblib.dump(clf_cell, 'models/classifier_cell_type.pkl')
        joblib.dump(clf_ery, 'models/classifier_erythrocyte.pkl')
        joblib.dump(clf_ley, 'models/classifier_leukocyte.pkl')
        print('Модели успешно обучены и сохранены.')
