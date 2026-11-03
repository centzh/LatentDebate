from pathlib import Path

from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch
import json

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs, output_hidden_states=True)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
predicted_class = model.config.id2label[predicted_class_idx]
print("Predicted class:", predicted_class)

cat_class = 'tiger cat'
dog_class = 'golden retriever'
cat_idx = model.config.label2id[cat_class]
dog_idx = model.config.label2id[dog_class]

num_tokens = outputs.hidden_states[0].shape[1]

initial_strengths = torch.zeros(num_tokens, len(model.vit.layers))
# Note: This does not include dropout layer, only ViT layers
for token in range(num_tokens):
    for layer in range(len(model.vit.layers)):
        # Number of tokens is 197 (CLS + number of patches, given by HW/P^2 where P is patch size)
        token_activations = outputs.hidden_states[layer][0][token] # (768 dim vector, where embedding_dim = 768)
        interpreted_activations = model.classifier(token_activations) # Apply unembedding matrix (classifier head), mapping 768 dim vector to 1000 dim vector, representing 1000 classes in ImageNet

        cat_raw = interpreted_activations[cat_idx].unsqueeze(0)
        dog_raw = interpreted_activations[dog_idx].unsqueeze(0)
        cat_dog_raw = torch.cat((cat_raw, dog_raw))
        softmax_probs = torch.softmax(cat_dog_raw, dim=0)
        initial_strength = softmax_probs[0]
        initial_strengths[token][layer] = initial_strength

    print(f"Finished token: {token}")

data = {
        "label": 1, 
        "result": {
            "query": {
                "query":url, 
                "query_probs":initial_strengths.detach().cpu().tolist(),
                "token_score":0
            },
            "answer": {
                "answer": predicted_class,
                "tokenized_answer": predicted_class
            }
        }
    }

file = "data/cat_dog/vit_prob.jsonl"
output_path = Path(file)
output_path.parent.mkdir(parents=True, exist_ok=True)
with output_path.open("w", encoding="utf-8") as f:
    f.write(json.dumps(data) + "\n")
print(f"Completed writing initial strengths to {file}.")