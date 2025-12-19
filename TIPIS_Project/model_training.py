import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Загрузка данных
print("=" * 60)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 60)
df = pd.read_csv('bodyPerformance.csv')
print(f"Данные загружены: {df.shape[0]} строк, {df.shape[1]} столбцов")

# 1. Анализ целевой переменной
print("\n" + "=" * 60)
print("АНАЛИЗ ЦЕЛЕВОЙ ПЕРЕМЕННОЙ")
print("=" * 60)
print(df['class'].value_counts())
print("\nРаспределение классов:")
print(df['class'].value_counts(normalize=True).round(3))

# 2. Кодирование категориальных признаков
print("\n" + "=" * 60)
print("КОДИРОВАНИЕ ПРИЗНАКОВ")
print("=" * 60)

# Кодируем целевую переменную
target_encoder = LabelEncoder()
df['class_encoded'] = target_encoder.fit_transform(df['class'])
print(f"Целевая переменная закодирована: {dict(zip(target_encoder.classes_, range(len(target_encoder.classes_))))}")

# Кодируем пол (gender) если он есть
if 'gender' in df.columns:
    # Простое кодирование: M=0, F=1
    df['gender_encoded'] = df['gender'].map({'M': 0, 'F': 1})
    print("Пол закодирован: M=0, F=1")

# 3. Подготовка признаков и целевой переменной
print("\n" + "=" * 60)
print("ПОДГОТОВКА ДАННЫХ")
print("=" * 60)

# Определяем признаки для модели
# Включаем все числовые колонки, кроме целевой переменной и исходного пола
excluded_cols = ['class', 'class_encoded']

# Если есть исходная колонка gender, удаляем ее
if 'gender' in df.columns:
    excluded_cols.append('gender')

# Формируем список признаков
feature_columns = [col for col in df.columns if col not in excluded_cols]

# Проверяем типы данных
print("Типы данных в признаках:")
print(df[feature_columns].dtypes)

# Убедимся, что все признаки числовые
non_numeric = df[feature_columns].select_dtypes(exclude=[np.number]).columns
if len(non_numeric) > 0:
    print(f"Обнаружены нечисловые колонки: {list(non_numeric)}")
    # Удаляем нечисловые колонки
    feature_columns = [col for col in feature_columns if col not in non_numeric]
    print(f"Удалены нечисловые колонки. Осталось признаков: {len(feature_columns)}")

X = df[feature_columns]
y = df['class_encoded']

print(f"\nКоличество признаков: {len(feature_columns)}")
print("Признаки модели:")
for i, col in enumerate(feature_columns, 1):
    print(f"  {i:2}. {col}")

# 4. Разделение данных
print("\n" + "=" * 60)
print("РАЗДЕЛЕНИЕ ДАННЫХ")
print("=" * 60)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"Обучающая выборка: {X_train.shape}")
print(f"Тестовая выборка: {X_test.shape}")

# 5. Обучение модели XGBoost
print("\n" + "=" * 60)
print("ОБУЧЕНИЕ МОДЕЛИ XGBOOST")
print("=" * 60)
model = xgb.XGBClassifier(
    n_estimators=200,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    objective='multi:softprob',
    num_class=len(target_encoder.classes_)
)

# Кросс-валидация
print("ВЫПОЛНЕНИЕ КРОСС-ВАЛИДАЦИИ...")
cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
print(f"Результаты кросс-валидации (5 folds):")
print(f"  Средняя точность: {cv_scores.mean():.4f}")
print(f"  Стандартное отклонение: {cv_scores.std():.4f}")
print(f"  Диапазон: {cv_scores.min():.4f} - {cv_scores.max():.4f}")

# Обучение на полном тренировочном наборе
model.fit(X_train, y_train)
print("Модель обучена!")

# 6. Оценка модели
print("\n" + "=" * 60)
print("ОЦЕНКА МОДЕЛИ НА ТЕСТОВОЙ ВЫБОРКЕ")
print("=" * 60)
y_pred = model.predict(X_test)
y_pred_proba = model.predict_proba(X_test)

# Точность
accuracy = accuracy_score(y_test, y_pred)
print(f"Точность модели: {accuracy:.4f}")

# Детальный отчет
print("\nДЕТАЛЬНЫЙ ОТЧЕТ КЛАССИФИКАЦИИ:")
print(classification_report(y_test, y_pred, target_names=target_encoder.classes_))

# 7. Важность признаков
print("\n" + "=" * 60)
print("ВАЖНОСТЬ ПРИЗНАКОВ")
print("=" * 60)
feature_importance = pd.DataFrame({
    'feature': feature_columns,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

print("Топ-10 важнейших признаков:")
print(feature_importance.head(10).to_string(index=False))

# Визуализация важности признаков
plt.figure(figsize=(12, 8))
bars = plt.barh(feature_importance.head(15)['feature'][::-1], 
                feature_importance.head(15)['importance'][::-1])
plt.xlabel('Важность признака')
plt.title('Топ-15 важнейших признаков для классификации физической подготовки')
plt.tight_layout()

# Добавляем значения на столбцы
for bar in bars:
    width = bar.get_width()
    plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
             f'{width:.3f}', ha='left', va='center', fontsize=9)

plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
print("\nГрафик важности признаков сохранен как 'feature_importance.png'")

# 8. Сохранение модели и вспомогательных файлов
print("\n" + "=" * 60)
print("СОХРАНЕНИЕ МОДЕЛИ И ФАЙЛОВ")
print("=" * 60)

# Сохраняем модель
with open('xgboost_model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Сохраняем список признаков
with open('feature_columns.pkl', 'wb') as f:
    pickle.dump(feature_columns, f)

# Сохраняем кодировщик целевой переменной
with open('target_encoder.pkl', 'wb') as f:
    pickle.dump(target_encoder, f)

print("Файлы сохранены:")
print("   - xgboost_model.pkl (модель)")
print("   - feature_columns.pkl (список признаков)")
print("   - target_encoder.pkl (кодировщик классов)")

# 9. Пример предсказания
print("\n" + "=" * 60)
print("ПРИМЕР ПРЕДСКАЗАНИЯ")
print("=" * 60)
sample_idx = np.random.randint(0, len(X_test))
sample_features = X_test.iloc[sample_idx:sample_idx+1]
true_class = target_encoder.inverse_transform([y_test.iloc[sample_idx]])[0]
pred_class = target_encoder.inverse_transform(model.predict(sample_features))[0]
pred_proba = model.predict_proba(sample_features)[0]

print(f"Истинный класс: {true_class}")
print(f"Предсказанный класс: {pred_class}")
print("Вероятности по классам:")
for class_name, prob in zip(target_encoder.classes_, pred_proba.round(3)):
    print(f"  {class_name}: {prob:.3f}")

print("\n" + "=" * 60)
print("ОБУЧЕНИЕ ЗАВЕРШЕНО!")
print("=" * 60)