import streamlit as st
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.utils import load_model_and_mappings, predict_image
from config import config

st.set_page_config(page_title="Классификатор птиц", page_icon="🐦", layout="wide")
st.title("🐦 Классификатор птиц по изображениям")
st.markdown("Загрузите изображение птицы, и модель определит её вид.")

@st.cache_resource
def load_model():
    try:
        model, mapping = load_model_and_mappings()
        transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                               [0.229, 0.224, 0.225])
        ])
        return model, transform, mapping
    except Exception as e:
        st.error(f"Ошибка загрузки модели: {e}")
        return None, None, None

model, transform, mapping = load_model()
if model is None:
    st.warning("Модель не загружена. Запустите download_data.py и train_model.py")
    st.stop()

with st.expander("Информация о модели"):
    st.markdown(f"""
    **Архитектура:** {mapping.get('model_name', 'simple_cnn')}
    **Размер изображения:** {config.image_size}x{config.image_size}
    **Классы:** {len(mapping['class_names'])}
    """)

col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("Загрузка изображения")
    option = st.radio("Способ загрузки:", ["Загрузить файл", "Использовать пример"])
    image = None
    if option == "Загрузить файл":
        uploaded_file = st.file_uploader("Выберите изображение...", type=['jpg', 'jpeg', 'png', 'bmp'])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert('RGB')
    elif option == "Использовать пример":
        example_images = {
            "Кряква": "https://inaturalist-open-data.s3.amazonaws.com/photos/11796983/medium.jpg",
            "Галка": "https://static.inaturalist.org/photos/16229227/medium.jpg",
            "Домашний голубь": "https://static.inaturalist.org/photos/12697102/medium.jpg",
            "Серая ворона": "https://static.inaturalist.org/photos/8273890/medium.jpg",
            "Серебристая чайка": "https://static.inaturalist.org/photos/21701448/medium.jpg"
        }
        selected = st.selectbox("Выберите пример:", list(example_images.keys()))
        if st.button("Загрузить пример"):
            import requests
            from io import BytesIO
            try:
                response = requests.get(example_images[selected], timeout=10)
                image = Image.open(BytesIO(response.content)).convert('RGB')
                st.success(f"Загружен пример: {selected}")
            except Exception as e:
                st.error(f"Ошибка загрузки: {e}")

with col2:
    if image is not None:
        st.subheader("Результаты")
        st.image(image, caption="Изображение", use_column_width=True)
        with st.spinner('Анализ...'):
            predictions, top_confidence = predict_image(image, model, transform, mapping)
        st.success(f"**Топ-1 уверенность: {top_confidence:.1f}%**")
        for i, pred in enumerate(predictions, 1):
            st.markdown(f"**{i}. {pred['species']}** — {pred['confidence']:.1f}%")
            st.progress(pred['confidence'] / 100)
    else:
        st.info("Загрузите изображение")