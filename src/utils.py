import json
import torch
from pathlib import Path

def load_model_and_mappings(model_path="models/best_model.pth",
                           mapping_path="data/processed/species_mapping.json"):
    if not Path(model_path).exists():
        raise FileNotFoundError(f"Модель не найдена: {model_path}")
    if not Path(mapping_path).exists():
        raise FileNotFoundError(f"Маппинг не найден: {mapping_path}")
    
    with open(mapping_path, 'r') as f:
        mapping = json.load(f)
    
    checkpoint = torch.load(model_path, map_location='cpu')
    model_name = checkpoint.get('model_name', 'simple_cnn')
    
    from src.model_training import BirdClassifier
    model = BirdClassifier(
        num_classes=len(mapping['class_names']),
        model_name=model_name
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, mapping

def predict_image(image, model, transform, mapping, top_k=3):
    image_tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, 1)
    class_idx = predicted.item()
    confidence_score = confidence.item() * 100
    top_k_prob, top_k_idx = torch.topk(probabilities, top_k)
    predictions = []
    for prob, idx in zip(top_k_prob[0], top_k_idx[0]):
        species_key = str(idx.item())
        species_name = mapping['idx_to_species'].get(species_key, f"Class {idx.item()}")
        predictions.append({
            'species': species_name,
            'confidence': prob.item() * 100,
            'label': idx.item()
        })
    return predictions, confidence_score