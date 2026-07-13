import os
import io
import json
import base64
import logging
from typing import Tuple, List, Dict, Any, Optional

import numpy as np
import tensorflow as tf
from PIL import Image
from groq import Groq
from dotenv import load_dotenv

import matplotlib
matplotlib.use('Agg')  # Thread-safe headless backend for web servers
import matplotlib.pyplot as plt
import seaborn as sns

from django.shortcuts import render, redirect
from django.core.files.storage import FileSystemStorage
from django.core.files.uploadedfile import UploadedFile
from django.http import HttpRequest, HttpResponse
from django.conf import settings
from django.utils import timezone
from django.db.models import Count
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required

from .models import Prediction

# Setup enterprise logging
logger = logging.getLogger(__name__)
load_dotenv()

# =========================
# CONFIGURATION & CONSTANTS
# =========================
MODEL_PATH = getattr(settings, "PLANT_MODEL_PATH", "detector/model/plant_disease_model.h5")
IMG_SIZE = (128, 128)
ALLOWED_EXTENSIONS = {"jpg", "jpeg", "png"}
MAX_FILE_SIZE = 5 * 1024 * 1024  # 5MB

CLASS_NAMES = [
    'Corn Common Rust', 'Corn Healthy', 'Grape Black Rot',
    'Potato Early Blight', 'Potato Healthy', 'Potato Late Blight',
    'Tomato Early Blight', 'Tomato Healthy', 'Tomato Late Blight',
    'Tomato Leaf Mold'
]

PALETTE = ["#22c55e", "#3b82f6", "#f59e0b", "#ef4444", "#8b5cf6", "#14b8a6"]

# Global lazy-loaded instances
_MODEL: Optional[tf.keras.Model] = None
_GROQ_CLIENT: Optional[Groq] = None


# =========================
# HELPER INITIALIZERS
# =========================
def get_model() -> tf.keras.Model:
    """Lazy-loads the Keras model with error isolation."""
    global _MODEL
    if _MODEL is None:
        if not os.path.exists(MODEL_PATH):
            logger.critical(f"Model file not found at path: {MODEL_PATH}")
            raise FileNotFoundError(f"Model missing at {MODEL_PATH}")
        try:
            # compile=False optimizes load time and memory footprint
            _MODEL = tf.keras.models.load_model(MODEL_PATH, compile=False)
            logger.info("TensorFlow plant disease model loaded successfully.")
        except Exception as e:
            logger.exception("Failed to initialize TensorFlow model.")
            raise e
    return _MODEL


def get_groq_client() -> Optional[Groq]:
    """Lazy-loads the Groq API Client."""
    global _GROQ_CLIENT
    if _GROQ_CLIENT is None:
        api_key = os.environ.get("GROQ_API_KEY")
        if api_key:
            _GROQ_CLIENT = Groq(api_key=api_key)
        else:
            logger.warning("GROQ_API_KEY environment variable is not set.")
    return _GROQ_CLIENT


# =========================
# ML PIPELINE UTILITIES
# =========================
def preprocess_image(file_obj: Any) -> np.ndarray:
    """Transforms raw image into standardized tensor array."""
    with Image.open(file_obj) as img:
        img = img.convert("RGB").resize(IMG_SIZE)
        img_array = np.array(img, dtype=np.float32) / 255.0
        return np.expand_dims(img_array, axis=0)


def predict_disease(img_batch: np.ndarray) -> Tuple[str, float, List[Dict[str, Any]]]:
    """Runs inference on the image batch, returning top class and top-3 breakdown."""
    model = get_model()
    predictions = model.predict(img_batch, verbose=0)
    probabilities = np.squeeze(predictions)

    top_indices = np.argsort(probabilities)[::-1][:3]
    primary_idx = top_indices[0]
    
    confidence = round(float(probabilities[primary_idx]) * 100, 2)
    predicted_label = CLASS_NAMES[primary_idx]

    top_3_breakdown = [
        {
            "label": CLASS_NAMES[i],
            "confidence": round(float(probabilities[i]) * 100, 2)
        }
        for i in top_indices
    ]

    return predicted_label, confidence, top_3_breakdown


def get_ai_recommendation(label: str) -> dict:
    """Returns a structured dictionary with guaranteed normalized lowercase keys."""
    client = get_groq_client()
    
    # Safe default fallback structure
    fallback = {
        "cause": "Information is currently updating. Please try again.",
        "treatment": "Information is currently updating. Please try again.",
        "prevention": "Information is currently updating. Please try again.",
        "fertilizer": "Information is currently updating. Please try again."
    }
    
    if not client:
        return fallback

    system_instruction = (
        "You are an expert agricultural scientist. You must respond ONLY with a valid JSON object. "
        "Do not wrap the response in markdown code blocks or backticks. Use exactly these lowercase keys: "
        "'cause', 'treatment', 'prevention', 'fertilizer'."
    )

    prompt = f"""
Plant Disease: {label}

Provide a JSON object with exactly these four keys. Values must be short bullet points for farmers:
{{
    "cause": "- Explain what biological agent (fungus/bacteria) or weather caused this and explain simply about the fungus/bacteria in details paragraph and points in different Sytematically",
    "treatment": "- List specific chemical or organic sprays to cure it right now in in details  paragrapgh and points in different Sytematically",
    "prevention": "- List long-term practices to avoid it in the future in details paragraph and points in different Sytematically",
    "fertilizer": "- Provide nutrient recovery or fertilizers tips in details paragraph and points in different Sytematically"
}}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[
                {"role": "system", "content": system_instruction},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"},
            temperature=0.1,
            timeout=10.0
        )
        
        raw_content = response.choices[0].message.content.strip()
        
        # Defensive Fix: Remove any markdown wrapper backticks if the model accidentally added them
        if raw_content.startswith("```"):
            raw_content = raw_content.strip("```").strip("json").strip()
            
        raw_data = json.loads(raw_content)
        
        # Case Insensitivity Normalization: Convert all keys to lowercase to prevent Django template mismatch
        normalized_data = {str(k).lower().strip(): v for k, v in raw_data.items()}
        
        # Ensure all expected keys exist with clean fallbacks if missing
        return {
            "cause": normalized_data.get("cause", fallback["cause"]),
            "treatment": normalized_data.get("treatment", fallback["treatment"]),
            "prevention": normalized_data.get("prevention", fallback["prevention"]),
            "fertilizer": normalized_data.get("fertilizer", fallback["fertilizer"]),
        }
        
    except Exception as e:
        logger.exception(f"Groq structural payload parsing failed for: {label}")
        return fallback


def generate_base64_chart(fig: plt.Figure) -> str:
    """Converts a Matplotlib figure straight into a web-safe Base64 String (No Disk Write)."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=180, facecolor='#0f172a', bbox_inches='tight')
    buf.seek(0)
    string = base64.b64encode(buf.read()).decode('utf-8')
    plt.close(fig)
    return f"data:image/png;base64,{string}"


# =========================
# REQUEST HANDLERS (VIEWS)
# =========================
@login_required(login_url='login')
def home(request: HttpRequest) -> HttpResponse:
    context = {
        "image_url": None, "prediction": None, "confidence": None,
        "ai_response": None, "error_message": None, "prediction_top": None
    }

    if request.method == "POST" and request.FILES.get("image"):
        image: UploadedFile = request.FILES["image"]
        try:
            # Strict Validations
            ext = os.path.splitext(image.name)[1].lower().lstrip('.')
            if ext not in ALLOWED_EXTENSIONS:
                raise ValueError("Unsupported format. Please upload a JPG, JPEG, or PNG image.")
            
            if image.size > MAX_FILE_SIZE:
                raise ValueError("The uploaded file exceeds the maximum security limit of 5MB.")

            # Write file securely to disk
            fs = FileSystemStorage()
            filename = fs.save(image.name, image)
            context["image_url"] = fs.url(filename)

            # Core ML Operations
            img_tensor = preprocess_image(image)
            prediction, confidence, top_3 = predict_disease(img_tensor)

            context.update({
                "prediction": prediction,
                "confidence": confidence,
                "prediction_top": top_3,
                "ai_response": get_ai_recommendation(prediction)
            })

            # Persistent DB Write
            Prediction.objects.create(
                image=filename,
                disease=prediction,
                confidence=confidence
            )

        except ValueError as val_err:
            context["error_message"] = str(val_err)
        except Exception:
            logger.exception("An unhandled exception occurred during image processing execution.")
            context["error_message"] = "An internal processing error occurred. Please try again."

    return render(request, "home.html", context)


@login_required(login_url='login')
def analytics(request: HttpRequest) -> HttpResponse:
    query_set = Prediction.objects.values('disease').annotate(total=Count('disease'))
    
    labels = [item['disease'] for item in query_set]
    values = [item['total'] for item in query_set]

    total_predictions = sum(values)
    disease_types = len(labels)

    # UI Context fallback parameters
    chart_labels = labels if labels else ['No Records Available']
    chart_values = values if values else [1]

    sns.set_theme(style='darkgrid')

    # Chart 1: Matplotlib Bar Chart Execution
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(chart_values))]
    sns.barplot(x=chart_labels, y=chart_values, palette=colors, ax=ax1)
    ax1.set_title('Disease Detection Frequency', color='white', pad=15)
    ax1.set_xlabel('Disease', color='white')
    ax1.set_ylabel('Count', color='white')
    ax1.tick_params(colors='white')
    plt.xticks(rotation=25, ha='right')
    bar_chart_data = generate_base64_chart(fig1)

    # Chart 2: Matplotlib Doughnut Pie Execution
    fig2, ax2 = plt.subplots(figsize=(7, 5))
    wedges, _ = ax2.pie(chart_values, colors=colors, startangle=90, wedgeprops={'edgecolor': '#0f172a'})
    centre_circle = plt.Circle((0, 0), 0.6, fc='#0f172a')
    ax2.add_artist(centre_circle)
    ax2.set_title('Disease Distribution', color='white', pad=15)
    doughnut_chart_data = generate_base64_chart(fig2)

    return render(request, "analytics.html", {
        "labels": json.dumps(labels),
        "values": json.dumps(values),
        "total_predictions": total_predictions,
        "disease_types": disease_types,
        "bar_chart_url": bar_chart_data,       # Simply swap `<img src="{{ bar_chart_url }}">` in template!
        "doughnut_chart_url": doughnut_chart_data,
    })


# =========================
# AUTHENTICATION & STATIC
# =========================
def features(request: HttpRequest) -> HttpResponse:
    return render(request, "features.html")


def contact(request: HttpRequest) -> HttpResponse:
    return render(request, "contact.html")


def signup(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        form = UserCreationForm(request.POST)
        if form.is_valid():
            form.save()
            return redirect("login")
    else:
        form = UserCreationForm()

    return render(request, "registration/signup.html", {"form": form})


def redirect_to_login(request: HttpRequest) -> HttpResponse:
    """Fixed Indentation Defect"""
    return redirect('login')