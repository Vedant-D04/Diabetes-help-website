import os
import tempfile
from typing import List

import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch
from torchvision import transforms
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

def load_model(model_path: str = '/Users/bhargavdesai/Desktop/nmims/mini proj/model.pth'):
    try:
        model = model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()
        return model
    except Exception as e:
        print(f"Error loading the model: {e}")
        return None

model = load_model()

def load_image(image_path: str):
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError("Failed to load image")
        
        image = cv2.resize(image, (224, 224))
        image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), 224 / 10), -4, 128)
        image = Image.fromarray(image)
        
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])
        image = transform(image)
        return image.unsqueeze(0)
    except Exception as e:
        raise ValueError(f"Error processing image: {e}")

def load_diabetes_model(csv_path: str = 'diabetes.csv'):
    try:
        df = pd.read_csv(csv_path)
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        log_reg = LogisticRegression(max_iter=1000)
        log_reg.fit(X_train, y_train)
        return log_reg
    except Exception as e:
        print(f"Error loading diabetes dataset or training model: {e}")
        return None

log_reg = load_diabetes_model()

# Load a more advanced language model for the FAQBot
tokenizer = AutoTokenizer.from_pretrained("gpt2")
faqbot_model = AutoModelForCausalLM.from_pretrained("gpt2").to(device)

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

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
        data = np.array([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
        prediction = log_reg.predict(data)
        result = "Diabetes Detected" if prediction[0] == 1 else "No Diabetes Detected"
        return templates.TemplateResponse("detect_diabetes.html", {"request": request, "prediction": result})
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
        
        _, predicted = torch.max(output.logits, 1)

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

        print("Prediction:", result)
        print("Class Description:", class_description)

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
        # Prepare the input for the model
        prompt = f"Answer the following question about diabetes:\nQ: {user_question}\nA:"
        input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

        # Generate the answer
        with torch.no_grad():
            output = faqbot_model.generate(
                input_ids,
                max_length=150,
                num_return_sequences=1,
                no_repeat_ngram_size=2,
                top_k=50,
                top_p=0.95,
                temperature=0.7
            )

        answer = tokenizer.decode(output[0], skip_special_tokens=True)
        
        # Extract only the generated answer part
        answer = answer.split("A:")[-1].strip()

        return templates.TemplateResponse("faqbot.html", {
            "request": request, 
            "answer": answer,
            "user_question": user_question
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error during chatbot interaction: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)