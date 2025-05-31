import cv2
import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import tkinter as tk
from tkinter import ttk
from tkinter import PhotoImage
from tkinter import filedialog
from tkinter import Button
from tkinter import filedialog
from PIL import Image, ImageTk

clf_cell = joblib.load('models/classifier_cell_type.pkl')
clf_ery = joblib.load('models/classifier_erythrocyte.pkl')
clf_ley = joblib.load('models/classifier_leukocyte.pkl')
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

def show_diagnosis_info(subtype):
    info = {
        'microcytosis': 'Обнаружен микроцитоз. Размер эритроцитов в крови уменьшен. За подробностями перейдите в раздел «Справка».',
        'macrocytosis': 'Обнаружен макроцитоз. Размер эритроцитов в крови увеличен. За подробностями перейдите в раздел «Справка».',
        'norm': 'Патологий в клетках не обнаружено.',
        'leukocytosis': 'Обнаружено повышение количества лейкоцитов в крови. За подробностями перейдите в раздел «Справка».',
        'leukopenia': 'Обнаружено снижение количества лейкоцитов в крови. За подробностями перейдите в раздел «Справка».',
    }
    return info.get(subtype, 'Информация отсутствует.')
detailed_info = {
    'microcytosis': '''Микроцитоз — это состояние, при котором размеры эритроцитов меньше нормы.
• Размер эритроцитов уменьшается до менее 6 мкм.

• Возможные причины: железодефицитная анемия, талассемия и хронические заболевания.
• Уменьшенная поверхность эритроцитов снижает способность транспортировать кислород, что приводит к гипоксии тканей.
• Симптомы: усталость, одышка, головокружение и бледность кожи.

• Основные группы риска:
1. Женщины репродуктивного возраста
2. Дети раннего возраста
3. Пожилые люди
4. Пациенты с болезнями ЖКТ
5. Спортсмены
6. Строгие вегетарианцы и веганы
7. Курящие и алкоголики
8. Люди с хроническими инфекциями и воспалениями''',
    'macrocytosis': '''Микроцитоз — это состояние, при котором размеры эритроцитов меньше нормы.
• Размер эритроцитов увеличивается до более 9 мкм.

• Возможные причины: дефицит витамина B12, фолиевой кислоты, алкоголизм, заболевания печени и некоторыt формы анемий.
• Большие эритроциты менее гибки и могут застревать в капиллярах, вызывая нарушение кровообращения.
• Симптомы: может наблюдаться увеличение селезенки (спленомегалия), так как она удаляет аномально крупные клетки из кровотока.

• Основные группы риска:
1. Алкоголики
2. Пожилые люди
3. Пациенты с болезнью щитовидной железы
4. Лица с лекарственно-индуцированным макрокритозом
5. Больные с заболеваниями печени
6. Фертильные женщины
7. Обладатели редких генетических расстройств''',
    'norm': 'Клетки находятся в пределах нормы. Дополнительная информация не требуется.',
    'leukocytosis': '''Лейкоцитоз — повышенное содержание лейкоцитов в крови.
• Повышение количества лейкоцитов выше нормы.
Важно понимать, что при увеличении общего числа лейкоцитов сами клетки остаются практически тех же размеров,
поскольку изменения размеров отдельных лейкоцитов крайне редки и чаще всего связаны
с патологиями клеточных мембран или ядер.

• Возможные причины: инфекция, воспаление, стресс,
заболевания кроветворной системы (лейкемия, лимфома), реакция на прием лекарств.
• Последствия лейкоцитоза включают повышенную склонность к тромбообразованию,
снижение эффективности иммунной защиты и повышение риска осложнений хронических заболеваний.

• Основные группы риска:
1. Пациенты с ослабленным иммунитетом вследствие хронических болезней,
включая ВИЧ/СПИД, диабет, аутоиммунные расстройства и онкологические заболевания
2. Пожилые люди
3. Лица, перенесшие трансплантацию органов
4. Те, кто подвергается частым хирургическим вмешательствам или имеет тяжелые травмы
5. Курящие и алкоголики
6. Работники медицинских учреждений, военнослужащие и спасатели,
часто сталкивающиеся с потенциальными источниками инфекционных агентов''',
    'leukopenia': '''Лейкопения — сниженное содержание лейкоцитов.
Может быть вызвана вирусными инфекциями, аутоиммунными заболеваниями или приёмом лекарств.
• Размер эритроцитов увеличивается до более 9 мкм.

• Возможные причины: вирусные инфекции, аутоимунные заболевания, прием лекарств, авитаминозы,
лучевое воздействие и химиотерапия, тяжелые формы гепатита и цирроза печени.
• Возможные осложнения лейкопении включают повышенный риск инфекций, замедленное заживление ран и общее ослабление защитных функций организма.

• Основные группы риска:
1. Лица, принимающих лекарственные препараты,
способные подавлять производство лейкоцитов.
2. Пожилые люди
3. Лица с ослабленной иммунной системой
4. Те, кто подвергался воздействию высоких доз радиации или химических веществ,
повреждающих костный мозг.
5. Больные с заболеваниями печени
6. Пациенты, проходящие курсы химиотерапии или лучевой терапии
7. Лица с хроническими аутоиммунными заболеваниями,
такими как системная красная волчанка или ревматоидный артрит
8. Беременные женщины'''
}

def update_help_text(subtype):
    text = detailed_info.get(subtype, 'Информация по данной патологии отсутствует.')
    help_label.config(text=text)

def analyze_image(image_path, clf_cell, clf_ery, clf_ley):
    img = cv2.imread(image_path)
    features_and_contours = extract_features(image_path)
    if not features_and_contours:
        return None, 'Контуры не найдены.'

    results = []
    for features, cnt, color_type in features_and_contours:
        pred_cell_type = clf_cell.predict([features])[0]
        subtype = clf_ery.predict([features])[0] if pred_cell_type == 'erythrocyte' else clf_ley.predict([features])[0]
        results.append((pred_cell_type, subtype, features[0], cnt))

    dominant = max(results, key=lambda x: x[2])
    cell_type, subtype, _, best_cnt = dominant
    info = show_diagnosis_info(subtype)

    img_result = img.copy()
    cv2.drawContours(img_result, [best_cnt], -1, (0, 255, 0), 2)
    x, y, w, h = cv2.boundingRect(best_cnt)
    label = f'{cell_type} - {subtype}'
    cv2.putText(img_result, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    result_text = f'Тип клетки: {label}\n{info}'
    update_help_text(subtype)
    return img_result, result_text

c1 = '#F8F9FF'
c2 = '#1E40AF'
c3 = '#F0F0F0'
c4 = '#999999'
c5 = '#000000'

WINDOW_WIDTH = 1280
NOTEBOOK_WIDTH = 1120
WINDOW_HEIGHT = 720

W = tk.Tk()
W.title('Анализ кровяных клеток')
W.geometry(f'{WINDOW_WIDTH}x{WINDOW_HEIGHT}')
W.configure(bg='#F8F9FF')
        
title = tk.Label(W, text='Анализ кровяных клеток', font=('Arial', 24, 'bold'), bg=c1, fg=c2)
title.pack(pady=(20, 5))

subtitle = tk.Label(W, text='Определение эритроцитов, лейкоцитов и возможных патологий', font=('Arial', 12), bg=c1, fg=c4)
subtitle.pack(pady=(2,20))

notebook_frame = tk.Frame(W, width=NOTEBOOK_WIDTH, height=500, bg='white')
notebook_frame.pack(pady=(10,50))
notebook_frame.pack_propagate(False)

style = ttk.Style()
style.configure('My.TLabel',          
                font='Arial 14',    
                fg=c2,   
                padding=10,             
                bg=c3)   

style.configure('TNotebook.Tab', font=('Arial', 12), padding=[20, 10], background='lightgray')
style.map('TNotebook.Tab',
          background=[('selected', 'white')],
          foreground=[('selected', 'black')])

notebook = ttk.Notebook(notebook_frame)
notebook.pack(expand=True, fill='both')

icon_analysis = PhotoImage(file='icon1.png')
icon_pathology = PhotoImage(file='icon2.png')
icon_help = PhotoImage(file='icon3.png')

icons = [icon_analysis, icon_pathology, icon_help]

#Вкладки
tabs = []
tab_titles = ['Анализ', 'Результат', 'Справка']

for i, title_text in enumerate(tab_titles):
    tab = ttk.Frame(notebook)
    notebook.add(tab, text=title_text, image=icons[i], compound='left')  # compound='left' — иконка слева от текста
    tabs.append(tab)

notebook.image_refs = icons

#Вкладка "Анализ"
tab_analysis = tabs[0]

left_frame = tk.Frame(tab_analysis, bg=c1, width=NOTEBOOK_WIDTH//2, height=460)
left_frame.pack(side='left', fill='both', expand=False, padx=20, pady=20)
left_frame.pack_propagate(False)

right_frame = tk.Frame(tab_analysis, bg=c1, width=NOTEBOOK_WIDTH//2, height=460)
right_frame.pack(side='right', fill='both', expand=False, padx=20, pady=20)
right_frame.pack_propagate(False)

left_top = tk.Frame(left_frame, bg=c1)
left_top.pack(side='top', fill='x')

left_spacer = tk.Frame(left_frame, bg=c1)
left_spacer.pack(expand=True, fill='both')

left_bottom = tk.Frame(left_frame, bg=c1)
left_bottom.pack(side='bottom', fill='x')

icon5 = PhotoImage(file='icon5.png')
icon5_label = tk.Label(left_top, image=icon5, bg=c1)
icon5_label.pack(pady=(70,10))

def load_image():
    file_path = filedialog.askopenfilename(filetypes=[('Изображения', '*.png *.jpg *.jpeg *.bmp')])
    if file_path:
        print('Загружен файл:', file_path)
        img_result, result_text = analyze_image(file_path, clf_cell, clf_ery, clf_ley)

        if img_result is None:
            result_label.config(text='Контуры не найдены.')
            return

        h, w = img_result.shape[:2]
        max_w, max_h = 500, 400
        scale = min(max_w / w, max_h / h)
        resized = cv2.resize(img_result, (int(w * scale), int(h * scale)))

        img_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        img_tk = ImageTk.PhotoImage(img_pil)

        result_image_label.config(image=img_tk)
        result_image_label.image = img_tk

        result_text_label.config(text=result_text)

        notebook.select(1)

icon4 = PhotoImage(file='icon4.png')
load_button = tk.Button(left_bottom,
                        text='Загрузить изображение',
                        command=load_image,
                        image=icon4,
                        compound='left',
                        font=('Arial', 12),
                        bg=c2, fg='white',
                        padx=10, pady=5)
load_button.image = icon4
load_button.pack(anchor='center', pady=(10,150))

right_top = tk.Frame(right_frame, bg=c1)
right_top.pack(side='top', fill='x')

icon6 = PhotoImage(file='icon6.png')
icon6_label = tk.Label(right_top, image=icon6, bg=c1)
icon6_label.pack(pady=(70,10))

right_spacer = tk.Frame(right_frame, bg=c1)
right_spacer.pack(expand=True, fill='both')

info_text = '''Загрузите изображение мазка крови. 
Система выполнит автоматический анализ:
• Определит эритроциты и лейкоциты
• Выявит патологические изменения
• Выведет вероятные диагнозы'''

info_label = tk.Label(right_spacer, text=info_text, font=('Arial', 12), bg=c1, fg=c2,
                      justify='left', anchor='center')
info_label.pack(expand=True, fill='both', padx=10, pady=(10,120), anchor='center')

#Вкладка "Результат"
tab_pathology = tabs[1]

result_left = tk.Frame(tab_pathology, bg=c1, width=NOTEBOOK_WIDTH//2, height=460)
result_left.pack(side='left', fill='both', expand=False, padx=20, pady=20)
result_left.pack_propagate(False)

result_right = tk.Frame(tab_pathology, bg=c1, width=NOTEBOOK_WIDTH//2, height=460)
result_right.pack(side='right', fill='both', expand=False, padx=20, pady=20)
result_right.pack_propagate(False)

result_image_label = tk.Label(result_left, bg=c1)
result_image_label.pack(expand=True)

result_text_label = tk.Label(result_right, text='Здесь появится результат анализа.', font=('Arial', 12), bg=c1, fg=c2,
                             justify='left', anchor='n', wraplength=400)
result_text_label.pack(padx=10, pady=10, fill='both', expand=True)

#Вкладка "Справка"
tab_help = tabs[2]

help_text = 'Информация о патологиях появится здесь после анализа изображения.'

help_label = tk.Label(tab_help, text=help_text, font=('Arial', 12), bg=c3, fg=c2, justify='left', wraplength=800)
help_label.pack(padx=20, pady=20, anchor='nw')

style.configure('TNotebook', tabposition='n')

W.mainloop()
