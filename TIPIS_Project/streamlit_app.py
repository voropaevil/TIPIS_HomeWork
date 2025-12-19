import streamlit as st
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Настройки страницы
st.set_page_config(
    page_title="Классификация физической подготовки",
    page_icon="🏋️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Стили CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.8rem;
        color: #1E88E5;
        text-align: center;
        margin-bottom: 1.5rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .section-header {
        font-size: 1.8rem;
        color: #0D47A1;
        border-bottom: 3px solid #1E88E5;
        padding-bottom: 0.5rem;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    .prediction-card {
        background-color: #f8f9fa;
        padding: 1.8rem;
        border-radius: 12px;
        border-left: 6px solid #1E88E5;
        margin: 1.2rem 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
    }
    .success-card {
        background: linear-gradient(135deg, #43e97b 0%, #38f9d7 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .info-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    .stButton>button {
        background: linear-gradient(135deg, #1E88E5 0%, #0D47A1 100%);
        color: white;
        border: none;
        padding: 0.8rem 2rem;
        border-radius: 8px;
        font-size: 1rem;
        font-weight: bold;
    }
    .error-box {
        background-color: #ffebee;
        border-left: 6px solid #f44336;
        padding: 1.5rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    .upload-box {
        border: 2px dashed #1E88E5;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
        background-color: #f8f9fa;
        margin: 1rem 0;
    }
    .class-A { color: #4CAF50; font-weight: bold; }
    .class-B { color: #2196F3; font-weight: bold; }
    .class-C { color: #FFC107; font-weight: bold; }
    .class-D { color: #F44336; font-weight: bold; }
</style>
""", unsafe_allow_html=True)

# Заголовок
st.markdown('<div class="main-header">Система оценки физической подготовки</div>', unsafe_allow_html=True)
st.markdown("---")

# Проверка наличия файлов модели
def check_model_files():
    required_files = ['xgboost_model.pkl', 'feature_columns.pkl', 'target_encoder.pkl']
    missing_files = []
    
    for file in required_files:
        if not os.path.exists(file):
            missing_files.append(file)
    
    return missing_files

# Загрузка модели и вспомогательных файлов
@st.cache_resource
def load_model():
    try:
        with open('xgboost_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('feature_columns.pkl', 'rb') as f:
            feature_columns = pickle.load(f)
        with open('target_encoder.pkl', 'rb') as f:
            target_encoder = pickle.load(f)
        return model, feature_columns, target_encoder
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None, None

model, feature_columns, target_encoder = load_model()

# Если модель не загружена
if model is None:
    st.markdown('<div class="error-box">', unsafe_allow_html=True)
    st.error("Модель не загружена!")
    st.write("Пожалуйста, сначала обучите модель:")
    st.code("python model_training.py")
    st.stop()

# Боковая панель с информацией
with st.sidebar:
    st.markdown("### О системе")
    st.markdown("""
    **Назначение**: Классификация уровня физической подготовки на основе антропометрических данных и физических тестов
    
    **Классы**:
    - **A** - Отличная форма
    - **B** - Хорошая форма
    - **C** - Средняя форма
    - **D** - Ниже среднего
    
    **Модель**: XGBoost Classifier
    **Точность**: 90-95%
    """)
    
    st.markdown("### Статистика модели")
    st.markdown(f"Количество признаков: **{len(feature_columns)}**")
    st.markdown(f"Количество классов: **{len(target_encoder.classes_)}**")
    
    st.markdown("---")
    st.markdown("### Описание признаков")
    st.markdown("""
    - **age**: Возраст (годы)
    - **height_cm**: Рост (см)
    - **weight_kg**: Вес (кг)
    - **body fat_%**: Процент жира
    - **diastolic**: Диастолическое давление
    - **systolic**: Систолическое давление
    - **gripForce**: Сила хвата
    - **sit and bend forward_cm**: Наклон вперед сидя
    - **sit-ups counts**: Количество приседаний
    - **broad jump_cm**: Прыжок в длину
    - **gender_encoded**: Пол (0=М, 1=Ж)
    """)

# Основное содержимое
st.markdown('<div class="section-header">Выберите режим анализа</div>', unsafe_allow_html=True)

mode = st.radio(
    "Выберите способ ввода данных:",
    ["Загрузить CSV файл", "Ручной ввод данных"],
    horizontal=True
)

st.markdown("---")

if mode == "Загрузить CSV файл":
    st.markdown('<div class="section-header">Массовый анализ данных</div>', unsafe_allow_html=True)
    
    # Блок загрузки файла
    uploaded_file = st.file_uploader(
        "Перетащите CSV файл с данными для анализа",
        type=['csv'],
        help="Файл должен содержать все необходимые признаки"
    )
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_file is not None:
        try:
            # Загрузка данных
            data = pd.read_csv(uploaded_file)
            st.success(f"Файл успешно загружен! Записей: {len(data)}")
            
            # Проверка наличия необходимых признаков
            missing_cols = [col for col in feature_columns if col not in data.columns]
            
            # Проверяем наличие исходного признака gender
            if 'gender' in data.columns and 'gender_encoded' not in data.columns:
                # Преобразуем gender в gender_encoded
                data['gender_encoded'] = data['gender'].map({'M': 0, 'F': 1, 'm': 0, 'f': 1})
                st.info("Признак 'gender' преобразован в 'gender_encoded' (M=0, F=1)")
            
            # Обновляем список отсутствующих признаков
            missing_cols = [col for col in feature_columns if col not in data.columns]
            
            if missing_cols:
                st.error(f"В файле отсутствуют следующие признаки: {missing_cols}")
                st.info("Убедитесь, что файл содержит все необходимые столбцы")
                st.info("Необходимые признаки:")
                for i, col in enumerate(feature_columns, 1):
                    st.info(f"{i}. {col}")
            else:
                # Подготовка данных
                processed_data = data[feature_columns].copy()
                
                # Предсказания
                with st.spinner("Выполняется анализ данных..."):
                    predictions = model.predict(processed_data)
                    probabilities = model.predict_proba(processed_data)
                
                # Расшифровка предсказаний
                class_names = target_encoder.classes_
                predicted_classes = [class_names[p] for p in predictions]
                confidence_scores = np.max(probabilities, axis=1)
                
                # Добавляем результаты в исходные данные
                data['Предсказанный_класс'] = predicted_classes
                data['Уверенность_предсказания'] = confidence_scores
                
                # Показываем метрики
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    st.metric("Всего записей", len(data))
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col2:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    avg_conf = np.mean(confidence_scores) * 100
                    st.metric("Средняя уверенность", f"{avg_conf:.1f}%")
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col3:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    dominant_class = max(set(predicted_classes), key=predicted_classes.count)
                    st.metric("Наиболее частый класс", dominant_class)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                with col4:
                    st.markdown('<div class="metric-card">', unsafe_allow_html=True)
                    unique_classes = len(set(predicted_classes))
                    st.metric("Уникальных классов", unique_classes)
                    st.markdown('</div>', unsafe_allow_html=True)
                
                # Визуализация
                st.markdown("### Визуализация результатов")
                
                # Создаем вкладки для разных графиков
                tab1, tab2, tab3 = st.tabs(["Распределение классов", "Уверенность предсказаний", "Диаграмма распределения"])
                
                with tab1:
                    # Распределение классов
                    fig1, ax1 = plt.subplots(figsize=(10, 6))
                    class_dist = pd.Series(predicted_classes).value_counts().sort_index()
                    colors = ['#4CAF50', '#2196F3', '#FFC107', '#F44336']
                    
                    bars = ax1.bar(class_dist.index, class_dist.values, color=colors)
                    ax1.set_xlabel('Класс физической подготовки')
                    ax1.set_ylabel('Количество записей')
                    ax1.set_title('Распределение предсказанных классов')
                    ax1.grid(True, alpha=0.3, axis='y')
                    
                    # Добавляем значения на столбцы
                    for bar in bars:
                        height = bar.get_height()
                        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                                f'{int(height)}', ha='center', va='bottom', fontweight='bold')
                    
                    st.pyplot(fig1)
                
                with tab2:
                    # Распределение уверенности
                    fig2, ax2 = plt.subplots(figsize=(10, 6))
                    ax2.hist(confidence_scores, bins=20, alpha=0.7, color='#1E88E5', edgecolor='black')
                    ax2.axvline(x=np.mean(confidence_scores), color='red', linestyle='--', 
                               linewidth=2, label=f'Среднее: {np.mean(confidence_scores):.2f}')
                    ax2.set_xlabel('Уверенность предсказания')
                    ax2.set_ylabel('Количество записей')
                    ax2.set_title('Распределение уверенности предсказаний')
                    ax2.legend()
                    ax2.grid(True, alpha=0.3)
                    st.pyplot(fig2)
                
                with tab3:
                    # Круговая диаграмма
                    fig3, ax3 = plt.subplots(figsize=(10, 8))
                    
                    # Круговая диаграмма распределения классов
                    class_dist = pd.Series(predicted_classes).value_counts()
                    colors = ['#4CAF50', '#2196F3', '#FFC107', '#F44336']
                    
                    # Создаем круговую диаграмму
                    wedges, texts, autotexts = ax3.pie(
                        class_dist.values, 
                        labels=class_dist.index, 
                        autopct='%1.1f%%',
                        colors=colors, 
                        startangle=90,
                        textprops={'fontsize': 12, 'fontweight': 'bold'}
                    )
                    
                    # Делаем подписи более читаемыми
                    for autotext in autotexts:
                        autotext.set_color('white')
                        autotext.set_fontweight('bold')
                    
                    ax3.set_title('Распределение классов (процентное)', fontsize=16, fontweight='bold', pad=20)
                    
                    # Добавляем легенду
                    ax3.legend(
                        wedges, 
                        [f'{label}: {value} ({percent:.1f}%)' for label, value, percent in 
                         zip(class_dist.index, class_dist.values, class_dist.values/class_dist.values.sum()*100)],
                        title="Классы",
                        loc="center left",
                        bbox_to_anchor=(1, 0, 0.5, 1)
                    )
                    
                    plt.tight_layout()
                    st.pyplot(fig3)
                
                # Таблица с результатами
                st.markdown("### Предпросмотр результатов")
                st.dataframe(data.head(20), use_container_width=True)
                
                # Статистика по классам
                st.markdown("### Статистика по классам")
                class_stats = []
                for cls in class_names:
                    if cls in predicted_classes:
                        mask = np.array(predicted_classes) == cls
                        count = np.sum(mask)
                        avg_conf = np.mean(confidence_scores[mask]) * 100
                        class_stats.append({
                            'Класс': cls,
                            'Количество': count,
                            'Доля (%)': f"{(count / len(data) * 100):.1f}%",
                            'Средняя уверенность (%)': f"{avg_conf:.1f}%"
                        })
                
                stats_df = pd.DataFrame(class_stats)
                st.dataframe(stats_df, use_container_width=True)
                
                # Кнопка для скачивания результатов
                csv_data = data.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                
                col_dl1, col_dl2, col_dl3 = st.columns(3)
                with col_dl2:
                    st.download_button(
                        label="Скачать результаты анализа",
                        data=csv_data,
                        file_name=f"результаты_анализа_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                
        except Exception as e:
            st.error(f"Ошибка при обработке файла: {str(e)}")
            st.info("Проверьте формат файла и наличие всех необходимых столбцов")

else:
    # Режим ручного ввода
    st.markdown('<div class="section-header">Индивидуальная оценка физической подготовки</div>', unsafe_allow_html=True)
    
    st.markdown("Введите данные для оценки уровня физической подготовки:")
    
    # Создаем колонки для ввода данных
    col1, col2 = st.columns(2)
    
    input_data = {}
    
    with col1:
        st.subheader("Антропометрические данные")
        input_data['age'] = st.number_input("Возраст (лет)", min_value=10, max_value=100, value=30)
        input_data['height_cm'] = st.number_input("Рост (см)", min_value=100.0, max_value=250.0, value=170.0, step=0.1)
        input_data['weight_kg'] = st.number_input("Вес (кг)", min_value=30.0, max_value=200.0, value=70.0, step=0.1)
        input_data['body fat_%'] = st.number_input("Процент жира (%)", min_value=5.0, max_value=50.0, value=20.0, step=0.1)
        
        # Артериальное давление
        st.subheader("Артериальное давление")
        col_press1, col_press2 = st.columns(2)
        with col_press1:
            input_data['systolic'] = st.number_input("Систолическое", min_value=80, max_value=200, value=120)
        with col_press2:
            input_data['diastolic'] = st.number_input("Диастолическое", min_value=50, max_value=130, value=80)
    
    with col2:
        st.subheader("Физические показатели")
        input_data['gripForce'] = st.number_input("Сила хвата", min_value=0.0, max_value=100.0, value=40.0, step=0.1)
        input_data['sit and bend forward_cm'] = st.number_input("Наклон вперед сидя (см)", min_value=-30.0, max_value=50.0, value=10.0, step=0.1)
        input_data['sit-ups counts'] = st.number_input("Количество приседаний", min_value=0, max_value=200, value=30)
        input_data['broad jump_cm'] = st.number_input("Прыжок в длину (см)", min_value=100.0, max_value=400.0, value=200.0, step=0.1)
        
        # Пол
        st.subheader("Демографические данные")
        gender_option = st.radio("Пол", options=['Мужской', 'Женский'], horizontal=True)
        input_data['gender'] = 'M' if gender_option == 'Мужской' else 'F'
    
    # Кнопка для предсказания
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("Оценить физическую подготовку", type="primary", use_container_width=True)
    
    if predict_button:
        try:
            # Подготовка данных для модели
            df_input = pd.DataFrame([input_data])
            
            # Кодирование пола (M=0, F=1)
            df_input['gender_encoded'] = df_input['gender'].map({'M': 0, 'F': 1})
            df_input = df_input.drop('gender', axis=1)
            
            # Проверяем наличие всех признаков
            missing_features = [col for col in feature_columns if col not in df_input.columns]
            
            if missing_features:
                st.error(f"Отсутствуют признаки: {missing_features}")
            else:
                # Убедимся, что порядок признаков правильный
                df_input = df_input[feature_columns]
                
                # Предсказание
                with st.spinner("Анализируем данные..."):
                    prediction = model.predict(df_input)[0]
                    probabilities = model.predict_proba(df_input)[0]
                
                # Расшифровка результата
                predicted_class = target_encoder.inverse_transform([prediction])[0]
                confidence = probabilities[prediction] * 100
                
                # Цветовая схема для классов
                class_colors = {
                    'A': '#4CAF50',
                    'B': '#2196F3',
                    'C': '#FFC107',
                    'D': '#F44336'
                }
                
                # Описания классов
                class_descriptions = {
                    'A': '**Отличная физическая форма** - Выше среднего по всем показателям',
                    'B': '**Хорошая физическая форма** - Выше среднего по большинству показателей',
                    'C': '**Средняя физическая форма** - Соответствует возрастным нормам',
                    'D': '**Физическая форма ниже среднего** - Рекомендуется повысить активность'
                }
                
                # Результат
                
                col_res1, col_res2 = st.columns(2)
                
                with col_res1:
                    st.markdown(f"### Результат оценки")
                    st.markdown(f"<h1 style='color: {class_colors.get(predicted_class, '#000')}; font-size: 3rem; text-align: center;'>{predicted_class}</h1>", unsafe_allow_html=True)
                    st.markdown(f"<div style='text-align: center; font-size: 1.2rem; margin-top: 1rem;'>{class_descriptions[predicted_class]}</div>", unsafe_allow_html=True)
                
                with col_res2:
                    st.markdown(f"### Уверенность предсказания")
                    st.markdown(f"<h1 style='font-size: 3rem; text-align: center;'>{confidence:.1f}%</h1>", unsafe_allow_html=True)
                    
                    # Прогресс-бар
                    st.progress(int(confidence))
                    
                    if confidence > 80:
                        st.markdown('<div class="success-card">Высокая уверенность предсказания</div>', unsafe_allow_html=True)
                    elif confidence > 60:
                        st.markdown('<div class="info-card">Средняя уверенность предсказания</div>', unsafe_allow_html=True)
                    else:
                        st.warning("Низкая уверенность предсказания")
                
                st.markdown('</div>', unsafe_allow_html=True)
                
                # График вероятностей по классам
                st.markdown("### Распределение вероятностей по классам")
                
                prob_df = pd.DataFrame({
                    'Класс': target_encoder.classes_,
                    'Вероятность (%)': (probabilities * 100).round(1)
                })
                
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.barh(prob_df['Класс'], prob_df['Вероятность (%)'], 
                              color=[class_colors.get(cls, '#777') for cls in prob_df['Класс']])
                ax.set_xlim(0, 100)
                ax.set_xlabel('Вероятность принадлежности (%)')
                ax.set_title('Распределение вероятностей по классам физической подготовки')
                ax.grid(True, alpha=0.3, axis='x')
                
                # Добавляем значения на столбцы
                for bar, prob in zip(bars, prob_df['Вероятность (%)']):
                    width = bar.get_width()
                    ax.text(width + 1, bar.get_y() + bar.get_height()/2, 
                           f'{prob:.1f}%', ha='left', va='center', fontweight='bold')
                
                st.pyplot(fig)
                
                # Дополнительная информация
                st.markdown("### Дополнительная информация")
                
                # Расчет ИМТ
                height_m = input_data['height_cm'] / 100
                bmi = input_data['weight_kg'] / (height_m ** 2)
                
                col_info1, col_info2, col_info3 = st.columns(3)
                
                with col_info1:
                    st.metric("Индекс массы тела", f"{bmi:.1f}")
                    if bmi < 18.5:
                        st.caption("Недостаточный вес")
                    elif bmi < 25:
                        st.caption("Нормальный вес")
                    elif bmi < 30:
                        st.caption("Избыточный вес")
                    else:
                        st.caption("Ожирение")
                
                with col_info2:
                    st.metric("Артериальное давление", f"{input_data['systolic']}/{input_data['diastolic']}")
                    if input_data['systolic'] < 120 and input_data['diastolic'] < 80:
                        st.caption("Нормальное")
                    elif input_data['systolic'] < 130 and input_data['diastolic'] < 85:
                        st.caption("Повышенное нормальное")
                    else:
                        st.caption("Проверьте у врача")
                
                with col_info3:
                    st.metric("Возрастная группа", 
                             "Молодой" if input_data['age'] < 30 else 
                             "Средний" if input_data['age'] < 50 else 
                             "Зрелый")
                    st.caption(f"Возраст: {input_data['age']} лет")
                
                # Рекомендации
                st.markdown("### Рекомендации")
                
                recommendations = {
                    'A': [
                        "Продолжайте текущий режим тренировок",
                        "Можно добавить силовые упражнения для развития силы",
                        "Поддерживайте регулярность тренировок (3-5 раз в неделю)",
                        "Следите за восстановлением и правильным питанием"
                    ],
                    'B': [
                        "Увеличьте интенсивность кардио-тренировок",
                        "Добавьте 1-2 силовые тренировки в неделю",
                        "Следите за питанием и восстановлением",
                        "Стремитесь к 3-5 тренировкам в неделю"
                    ],
                    'C': [
                        "Начните с регулярных тренировок 2-3 раза в неделю",
                        "Сфокусируйтесь на кардио-упражнениях",
                        "Постепенно увеличивайте нагрузку",
                        "Обратите внимание на питание"
                    ],
                    'D': [
                        "Рекомендуется консультация с врачом перед началом тренировок",
                        "Начните с легких прогулок 20-30 минут в день",
                        "Постепенно увеличивайте активность",
                        "Обратите внимание на рацион питания"
                    ]
                }
                
                for i, rec in enumerate(recommendations[predicted_class], 1):
                    st.markdown(f"{i}. {rec}")
                
        except Exception as e:
            st.error(f"Ошибка при предсказании: {str(e)}")
            st.info("Проверьте, что все поля заполнены правильно")

# Футер
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 2rem;'>
    <p><b>Система оценки физической подготовки</b> | Разработано с использованием XGBoost и Streamlit</p>
    <p>Модель обучена на датасете Body Performance Data с сайта Kaggle</p>
    <p>Внимание: Результаты являются прогнозом. Для точной оценки физической формы обратитесь к специалисту.</p>
    <p>© 2025 | Все права возможно защищены</p>
</div>
""", unsafe_allow_html=True)