import os
import json
import numpy as np
import tensorflow as tf
from PIL import Image

from groq import Groq
from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.http import HttpRequest
from dotenv import load_dotenv

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from django.conf import settings
from django.utils import timezone


from django.db.models import Count
from .models import Prediction

from django.contrib.auth.forms import UserCreationForm
from django.shortcuts import redirect

from django.contrib.auth.decorators import login_required




# =========================
# MODEL PATH
# =========================
MODEL_PATH = r"detector/model/plant_disease_model.h5"

_model = None
_model_load_error = None


# =========================
# LOAD MODEL (LAZY)
# =========================
def get_model():
    global _model, _model_load_error

    if _model is not None:
        return _model

    try:
        _model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        _model_load_error = str(e)
        _model = None

    return _model


# =========================
# CLASS NAMES
# =========================
class_names = [
    'Corn Common Rust',
    'Corn Healthy',
    'Grape Black Rot',
    'Potato Early Blight',
    'Potato Healthy',
    'Potato Late Blight',
    'Tomato Early Blight',
    'Tomato Healthy',
    'Tomato Late Blight',
    'Tomato Leaf Mold'
]


# =========================
# GROQ CLIENT
# =========================
load_dotenv()

def get_groq_client():
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        return None
    return Groq(api_key=api_key)


# =========================
# IMAGE PREPROCESS
# =========================
IMG_SIZE = (128, 128)

def preprocess_image(file_obj):
    img = Image.open(file_obj).convert("RGB")
    img = img.resize(IMG_SIZE)

    img = np.array(img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)

    return img


# =========================
# PREDICTION
# =========================
def predict_disease(img_batch):
    model = get_model()

    if model is None:
        raise RuntimeError("Model not loaded")

    pred = model.predict(img_batch)
    pred = np.squeeze(pred)

    index = int(np.argmax(pred))
    confidence = round(float(pred[index]) * 100, 2)

    top3 = np.argsort(pred)[::-1][:3]

    prediction_top = [
        {
            "label": class_names[i],
            "confidence": round(float(pred[i]) * 100, 2)
        }
        for i in top3
    ]

    return class_names[index], confidence, prediction_top


# =========================
# AI RECOMMENDATION
# =========================
def get_ai_recommendation(label):
    client = get_groq_client()

    if client is None:
        return "GROQ_API_KEY missing"

    prompt = f"""
You are an agriculture expert.

Disease: {label}

Give:
1. Cause
2. Treatment
3. Prevention
Simple language for farmers.
"""

    try:
        res = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
        )

        return res.choices[0].message.content

    except Exception as e:
        return f"AI Error: {str(e)}"


# =========================
# HOME VIEW (FIXED)
# =========================
@login_required(login_url='login')
def home(request: HttpRequest):

    image_url = None
    prediction = None
    confidence = None
    ai_response = None
    error_message = None
    prediction_top = None

    if request.method == "POST" and request.FILES.get("image"):
        try:
            image = request.FILES["image"]
            # FILE EXTENSION VALIDATION #
            
            allowed_extensions = ["jpg", "jpeg", "png"]
            ext = image.name.split(".")[-1].lower()
            if ext not in allowed_extensions:
                raise ValueError("Only JPG, JPEG, PNG files allowed")
            
            # FILE SIZE VALIDATION (5MB) #
            if image.size > 5 * 1024 * 1024:
                raise ValueError("Image size must be below 5MB")
            
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            image_url = fs.url(filename)

            img_batch = preprocess_image(image)
            prediction, confidence, prediction_top = predict_disease(img_batch)

            ai_response = get_ai_recommendation(prediction)

            # ✅ SAVE TO DATABASE (FIXED)
            Prediction.objects.create(
                image=filename,   # 🔥 IMPORTANT FIX
                disease=prediction,
                confidence=confidence
            )

        except Exception as e:
            error_message = "Something went wrong. Please try again."

    return render(request, "home.html", {
        "image_url": image_url,
        "prediction": prediction,
        "confidence": confidence,
        "ai_response": ai_response,
        "error_message": error_message,
        "prediction_top": prediction_top,
    })


# =========================
# ANALYTICS VIEW (FIXED)
# =========================

@login_required(login_url='login')
def analytics(request):

    data = Prediction.objects.values('disease').annotate(total=Count('disease'))
    labels = [x['disease'] for x in data]
    values = [x['total'] for x in data]

    total_predictions = int(sum(values))
    disease_types = len(labels)

    # Server-side (seaborn/matplotlib) chart generation
    # Saves to static/ so Django can serve them.
    chart_dir = os.path.join('static', 'generated')
    os.makedirs(chart_dir, exist_ok=True)

    # Make deterministic filenames; charts update when new data is generated.
    ts = int(timezone.now().timestamp())
    bar_path = os.path.join(chart_dir, f'analytics_bar_{ts}.png')
    doughnut_path = os.path.join(chart_dir, f'analytics_doughnut_{ts}.png')

    bar_url = f'/static/generated/{os.path.basename(bar_path)}'
    doughnut_url = f'/static/generated/{os.path.basename(doughnut_path)}'

    palette = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6"]

    # If no data, still render placeholder charts (prevents template logic complexity)
    if not labels or not values:
        labels = ['No data']
        values = [1]

    sns.set_theme(style='darkgrid')

    # --- Bar chart ---
    plt.figure(figsize=(10, 5))
    colors = [palette[i % len(palette)] for i in range(len(values))]
    sns.barplot(x=labels, y=values, palette=colors)
    plt.title('Disease Detection Frequency', color='white')
    plt.xlabel('Disease', color='white')
    plt.ylabel('Count', color='white')
    plt.xticks(rotation=30, ha='right', color='white')
    plt.yticks(color='white')
    plt.tight_layout()
    plt.savefig(bar_path, dpi=200, facecolor='#0f172a')
    plt.close()

    # --- Doughnut chart (pie with circle) ---
    plt.figure(figsize=(7, 5))
    colors = [palette[i % len(palette)] for i in range(len(values))]
    wedges, _texts = plt.pie(values, colors=colors, startangle=90, wedgeprops={'edgecolor': '#0f172a'})
    # cutout
    centre_circle = plt.Circle((0, 0), 0.6, fc='#0f172a')
    fig = plt.gcf()
    fig.gca().add_artist(centre_circle)
    plt.title('Disease Distribution', color='white')
    plt.tight_layout()
    plt.savefig(doughnut_path, dpi=200, facecolor='#0f172a')
    plt.close()

    return render(request, "analytics.html", {
        "labels": json.dumps([*labels]),
        "values": json.dumps([*values]),
        "total_predictions": total_predictions,
        "disease_types": disease_types,
        "bar_chart_url": bar_url,
        "doughnut_chart_url": doughnut_url,
    })


# =========================
# STATIC PAGES
# =========================
def features(request):
    return render(request, "features.html")


def contact(request):
    return render(request, "contact.html")

def signup(request):

    if request.method == "POST":

        form = UserCreationForm(request.POST)

        if form.is_valid():
            form.save()

            return redirect("login")

    else:
        form = UserCreationForm()

    return render(request, "registration/signup.html", {
        "form": form
    })
    
    
    def redirect_to_login(request):
        return redirect('login')