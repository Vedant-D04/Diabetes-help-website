import os
import tempfile
from typing import List
import joblib
import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import inception_v3, resnet50
from timm import create_model
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from fastapi import FastAPI, Request, Form, UploadFile, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from transformers import AutoTokenizer, AutoModelForCausalLM

app = FastAPI()
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the Hybrid Model architecture
class HybridModel(nn.Module):
    def __init__(self, num_classes=5):
        super(HybridModel, self).__init__()
        # Load pre-trained models
        self.inception = inception_v3(pretrained=True, aux_logits=True)
        self.resnet = resnet50(pretrained=True)
        self.vit = create_model('vit_base_patch16_224', pretrained=True)

        # Freeze the parameters of the ViT model
        for param in self.vit.parameters():
            param.requires_grad = False
            
        # Modify the classifier heads
        self.inception.fc = nn.Identity()
        self.resnet.fc = nn.Identity()
        
        # Combine the output features from all models
        self.fc = nn.Sequential(
            nn.Linear(5096, 1024),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, 100),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(100, num_classes),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x):
        # Resize input for InceptionV3
        inception_input = nn.functional.interpolate(x, size=(299, 299))
        
        # Forward pass through models
        if self.training:
            inception_out, _ = self.inception(inception_input)
        else:
            inception_out = self.inception(inception_input)
            
        resnet_out = self.resnet(x)
        vit_out = self.vit(x)
        
        # Concatenate the outputs
        combined = torch.cat((inception_out, resnet_out, vit_out), dim=1)
        
        # Pass through the final classifier
        output = self.fc(combined)
        return output

def load_model(model_path: str = 'model_v2'):
    try:
        model = HybridModel(num_classes=5)
        state_dict = torch.load(model_path, map_location=device)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        print("Hybrid model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading the hybrid model: {e}")
        return None

def load_image(image_path: str):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image")
        
        # Resize and preprocess the image
        image = cv2.resize(image, (224, 224))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 224 / 10), -4, 128)
        image = Image.fromarray(image)
        
        # Define the transformations
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        image = transform(image)
        return image.unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

def load_diabetes_model(model_path: str = '/Users/bhargavdesai/Desktop/nmims/mini proj/best_model.joblib'):
    try:
        model = joblib.load(model_path)
        print("Diabetes model loaded successfully.")
        return model
    except Exception as e:
        print(f"Error loading the diabetes model: {e}")
        return None

# Initialize models and tokenizer
model = load_model('model_v2')
log_reg = load_diabetes_model()
tokenizer = AutoTokenizer.from_pretrained("gpt2-medium")
faqbot_model = AutoModelForCausalLM.from_pretrained("gpt2-medium").to(device)

# Routes
@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.get("/about", response_class=HTMLResponse)
async def read_about(request: Request):
    return templates.TemplateResponse("about.html", {"request": request})

@app.get("/contact", response_class=HTMLResponse)
async def read_contact(request: Request):
    return templates.TemplateResponse("contact.html", {"request": request})

@app.get("/detect", response_class=HTMLResponse)
async def read_detect(request: Request):
    return templates.TemplateResponse("detect_diabetes.html", {"request": request})

@app.post("/predict")
async def predict(
    request: Request,
    Pregnancies: int = Form(...),
    Glucose: int = Form(...),
    BloodPressure: int = Form(...),
    SkinThickness: int = Form(...),
    Insulin: int = Form(...),
    BMI: float = Form(...),
    DiabetesPedigreeFunction: float = Form(...),
    Age: int = Form(...)
):
    if log_reg is None:
        raise HTTPException(status_code=500, detail="Diabetes prediction model not available")
    try:
        data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, 
                         Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction = log_reg.predict(data)
        result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes Detected"
        return templates.TemplateResponse(
            "detect_diabetes.html", 
            {"request": request, "prediction": result}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during prediction: {str(e)}")

@app.get("/detect_retinopathy", response_class=HTMLResponse)
async def read_detect_retinopathy(request: Request):
    return templates.TemplateResponse("detect_retinopathy.html", {"request": request})

@app.post("/detect_retinopathy")
async def detect_retinopathy(request: Request, image: UploadFile = Form(...)):
    if model is None:
        raise HTTPException(status_code=500, detail="Retinopathy detection model not available")
    
    try:
        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_file.write(image.file.read())
            temp_file_path = temp_file.name

        image = load_image(temp_file_path)
        image = image.to(device)

        with torch.no_grad():
            output = model(image)
        
        _, predicted = torch.max(output, 1)

        class_labels = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']
        result = class_labels[predicted.item()]
        predicted_class = class_labels.index(result)

        class_descriptions: List[str] = [
            "No Diabetic Retinopathy detected.",
            "Mild Diabetic Retinopathy detected. This is the earliest stage of diabetic retinopathy, and it may not cause any noticeable symptoms.",
            "Moderate Diabetic Retinopathy detected. This stage is characterized by the formation of microaneurysms, which are small, balloon-like swellings in the retina's tiny blood vessels.",
            "Severe Diabetic Retinopathy detected. This stage is characterized by the formation of new, fragile blood vessels in the retina, which can cause vision loss and blindness.",
            "Proliferative Diabetic Retinopathy detected. This is the most advanced stage of diabetic retinopathy, and it can cause severe vision loss and blindness."
        ]
        class_description = class_descriptions[predicted_class]

        return templates.TemplateResponse("detect_retinopathy.html", {
            "request": request,
            "prediction": result,
            "class_description": class_description
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during retinopathy detection: {str(e)}")
    finally:
        if 'temp_file_path' in locals():
            os.unlink(temp_file_path)

@app.get("/faqbot", response_class=HTMLResponse)
async def read_faqbot(request: Request):
    return templates.TemplateResponse("faqbot.html", {"request": request})

@app.post("/faqbot", response_class=HTMLResponse)
async def faqbot_response(request: Request, user_question: str = Form(...)):
    try:
        prompt = f"Answer the following question about diabetes:\nQ: {user_question}\nA:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            output = faqbot_model.generate(input_ids, max_length=250)
        
        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        return templates.TemplateResponse("faqbot.html", {
            "request": request,
            "answer": answer.replace(prompt, "").strip()
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating response: {str(e)}")
