import pandas as pd
import numpy as np

# Загружаем данные
df = pd.read_csv('bodyPerformance.csv')

print("=" * 50)
print("АНАЛИЗ ДАННЫХ О ФИЗИЧЕСКОЙ ПОДГОТОВКЕ")
print("=" * 50)

# 1. Основная информация
print("\n1. ОСНОВНАЯ ИНФОРМАЦИЯ:")
print(f"Размер данных: {df.shape}")
print(f"Количество строк: {df.shape[0]}")
print(f"Количество столбцов: {df.shape[1]}")

# 2. Названия столбцов
print("\n2. СТОЛБЦЫ:")
for i, col in enumerate(df.columns, 1):
    print(f"{i:2}. {col}")

# 3. Типы данных
print("\n3. ТИПЫ ДАННЫХ:")
print(df.dtypes)

# 4. Пропущенные значения
print("\n4. ПРОПУЩЕННЫЕ ЗНАЧЕНИЯ:")
missing = df.isnull().sum()
if missing.sum() > 0:
    print(missing[missing > 0])
else:
    print("Пропусков нет")

# 5. Целевая переменная
print("\n5. ЦЕЛЕВАЯ ПЕРЕМЕННАЯ (class):")
class_distribution = df['class'].value_counts()
print(class_distribution)
print(f"\nРаспределение (проценты):")
print(df['class'].value_counts(normalize=True).round(3))

# 6. Статистика по числовым признакам
print("\n6. СТАТИСТИКА ПО ЧИСЛОВЫМ ПРИЗНАКАМ:")
print(df.describe().T.round(2))

# 7. Корреляция
print("\n7. КОРРЕЛЯЦИЯ ПРИЗНАКОВ:")
numeric_cols = df.select_dtypes(include=[np.number]).columns
corr_matrix = df[numeric_cols].corr()

print("Наиболее коррелированные признаки (|корреляция| > 0.7):")
high_corr_found = False
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        corr_value = abs(corr_matrix.iloc[i, j])
        if corr_value > 0.7:
            print(f"{corr_matrix.columns[i]} - {corr_matrix.columns[j]}: {corr_value:.3f}")
            high_corr_found = True

if not high_corr_found:
    print("Нет сильно коррелированных признаков (|корреляция| <= 0.7)")