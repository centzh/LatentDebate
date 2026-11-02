from transformers import ViTImageProcessor, ViTForImageClassification
from PIL import Image
import requests
import torch

url = 'http://images.cocodataset.org/val2017/000000039769.jpg'
image = Image.open(requests.get(url, stream=True).raw)

processor = ViTImageProcessor.from_pretrained('google/vit-base-patch16-224')
model = ViTForImageClassification.from_pretrained('google/vit-base-patch16-224')

inputs = processor(images=image, return_tensors="pt")
outputs = model(**inputs, output_hidden_states=True)
logits = outputs.logits
# model predicts one of the 1000 ImageNet classes
predicted_class_idx = logits.argmax(-1).item()
print("Predicted class:", model.config.id2label[predicted_class_idx])

cat_class = 'tiger cat'
dog_class = 'golden retriever'
cat_idx = model.config.label2id[cat_class]
dog_idx = model.config.label2id[dog_class]

# Inspect hidden state value for a specific layer and token position
layer = 5
token = 0 

# Note: This does not include dropout layer, only ViT layers
for layer in range(len(model.vit.layers)):
    # Number of tokens is 197 (CLS + number of patches, given by HW/P^2 where P is patch size)
    token_activations = outputs.hidden_states[layer][0][token] # (768 dim vector, where embedding_dim = 768)
    interpreted_activations = model.classifier(token_activations) # Apply unembedding matrix (classifier head), mapping 768 dim vector to 1000 dim vector, representing 1000 classes in ImageNet

    cat_raw = interpreted_activations[cat_idx].unsqueeze(0)
    dog_raw = interpreted_activations[dog_idx].unsqueeze(0)
    cat_dog_raw = torch.cat((cat_raw, dog_raw))
    softmax_probs = torch.softmax(cat_dog_raw, dim=0)
    print(softmax_probs)


#  outputs.hidden_states[0].shape