import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================
# Configuration de la page
# ==========================
st.set_page_config(
    page_title="Dogs vs Cats Classifier",
    page_icon="🐶",
    layout="centered"
)

# ==========================
# Style CSS
# ==========================
st.markdown("""
<style>

.main {
    background-color: #f5f7fa;
}

h1{
    text-align:center;
    color:#2C3E50;
}

.subtitle{
    text-align:center;
    color:grey;
    font-size:20px;
}

.card{
    background:white;
    padding:30px;
    border-radius:20px;
    box-shadow:0px 6px 20px rgba(0,0,0,0.15);
}

.stButton>button{
    width:100%;
    border-radius:12px;
    background:#3b82f6;
    color:white;
    font-size:18px;
}

.stButton>button:hover{
    background:#2563eb;
    color:white;
}

</style>
""", unsafe_allow_html=True)

# ==========================
# Charger le modèle
# ==========================

model = tf.keras.models.load_model("best_model_dogs_cats.keras")

# ==========================
# Header
# ==========================

st.markdown("<h1>🐶 Dogs vs Cats Classifier 🐱</h1>", unsafe_allow_html=True)

st.markdown(
"""
<div class='subtitle'>
Deep Learning • CNN + Transfer Learning (MobileNetV2)
</div>
""",
unsafe_allow_html=True)

st.write("")

# ==========================
# Upload
# ==========================

uploaded_file = st.file_uploader(
    "📸 Choisissez une image",
    type=["jpg","jpeg","png"]
)

# ==========================
# Prediction
# ==========================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    col1,col2 = st.columns([1,1])

    with col1:

        st.image(
            image,
            caption="Image sélectionnée",
            use_container_width=True
        )

    img = image.resize((150,150))

    img = np.array(img)/255.0

    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)[0][0]

    with col2:

        st.subheader("Résultat")

        if prediction >=0.5:

            confidence = prediction

            st.success("🐶 Dog")

        else:

            confidence = 1-prediction

            st.success("🐱 Cat")

        st.metric(
            "Confiance",
            f"{confidence*100:.2f}%"
        )

        st.progress(float(confidence))

# ==========================
# Footer
# ==========================

st.markdown("---")

st.markdown(
"""
<center>

Développé avec ❤️ en utilisant

<b>TensorFlow • Keras • Streamlit</b>

</center>
""",
unsafe_allow_html=True
)