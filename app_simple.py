import streamlit as st
import numpy as np
import onnxruntime as ort
import json
from PIL import Image
import requests
from io import BytesIO

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Русские названия для научных имён (ключи — как в classes.json)
RU_NAMES = {
    "Anas_platyrhynchos":       "Кряква",
    "Coloeus_monedula":         "Галка",
    "Columba_livia_domestica":  "Домашний голубь",
    "Corvus_cornix":            "Серая ворона",
    "Larus_argentatus":         "Серебристая чайка",
}

def ru(species: str) -> str:
    """Возвращает русское название или исходное, если перевода нет."""
    return RU_NAMES.get(species, species)

@st.cache_resource
def load():
    session = ort.InferenceSession("bird_model.onnx", providers=["CPUExecutionProvider"])
    with open("classes.json", encoding="utf-8") as f:
        classes = json.load(f)
    return session, classes

def preprocess(img):
    img = img.convert("RGB").resize((224, 224))
    a = np.asarray(img, dtype=np.float32) / 255.0
    a = (a - MEAN) / STD
    return a.transpose(2, 0, 1)[None, ...].astype(np.float32)

def softmax(x):
    x = x - x.max(axis=1, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=1, keepdims=True)

st.set_page_config(page_title="Птицы", page_icon="🐦")
st.title("🐦 Классификатор птиц")

session, classes = load()

tab1, tab2 = st.tabs(["Файл", "Пример"])

img = None
with tab1:
    up = st.file_uploader("Картинка", type=["jpg", "jpeg", "png"])
    if up:
        img = Image.open(up)

with tab2:
    url = st.text_input(
        "URL картинки",
        "https://inaturalist-open-data.s3.amazonaws.com/photos/11796983/medium.jpg",
    )
    if st.button("Загрузить"):
        try:
            img = Image.open(BytesIO(requests.get(url, timeout=10).content))
        except Exception as e:
            st.error(f"Ошибка: {e}")

if img is not None:
    st.image(img, width=350)
    inp = session.get_inputs()[0].name
    out = session.run(None, {inp: preprocess(img)})[0]
    probs = softmax(out)[0]
    top = np.argsort(probs)[::-1][:3]

    # --- русские названия в выводе ---
    st.success(f"**{ru(classes[str(top[0])])}** — {probs[top[0]]*100:.1f}%")
    for i in top[1:]:
        st.write(f"{ru(classes[str(i)])} — {probs[i]*100:.1f}%")