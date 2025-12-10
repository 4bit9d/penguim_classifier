import os
import pickle
import numpy as np
import pandas as pd

def load_model(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Modelo não encontrado em: {path}")
    with open(path, "rb") as f:
        model = pickle.load(f)
    return model

# Este código assume que modelo tem predict() e predict_proba()
def predict_instance(model, x):
    """
    x: lista ou array com 4 floats
    retorna: (predicted_class_name, confidence, probabilities_array)
    """
    # Cria DataFrame com os mesmos nomes de colunas usados no treino
    df = pd.DataFrame([x], columns=[
        'culmen_length_mm',
        'culmen_depth_mm',
        'flipper_length_mm',
        'body_mass_g'
    ])
    
    probs = model.predict_proba(df)[0]  # probabilidades
    idx = int(probs.argmax())
    confidence = float(probs[idx])
    
    class_map = {0: "Adelie", 1: "Chinstrap", 2: "Gentoo"}
    
    try:
        label = model.classes_[idx]
        label_int = int(label)
        pred_name = class_map.get(label_int, str(label))
    except Exception:
        pred_name = str(model.predict(df)[0])
    
    return pred_name, confidence, probs