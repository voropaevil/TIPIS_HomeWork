# Проект: Классификация физической подготовки

data_analysis.py - Анализ данных (EDA), проверка структуры, корреляций
model_training.py - Обучение модели XGBoost с сохранением файлов
streamlit_app.py - Веб-приложение для работы с моделью
bodyPerformance.csv - Исходный датасет с физическими показателями
xgboost_model.pkl - Обученная модель XGBoost
feature_columns.pkl` - Список признаков для модели
target_encoder.pkl - Кодировщик классов (A,B,C,D → 0,1,2,3)
label_encoders.pkl - Кодировщики других признаков
feature_importance.png - График важности признаков
model_statistics.txt - Статистика работы модели
'data_info.txt - Информация о структуре данных
requirements.txt - Зависимости Python (установить через pip install)

# Как запустить проект
1. Установите зависимости: pip install -r requirements.txt
2. Обучите модель: python model_training.py
3. Запустите приложение: streamlit run streamlit_app.py

# Классификация
Модель определяет 4 уровня физической подготовки:
A - Отличная форма
B - Хорошая форма  
C - Средняя форма
D - Форма ниже среднего
