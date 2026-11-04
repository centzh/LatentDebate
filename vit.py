
from pathlib import Path

from transformers import ViTImageProcessor, ViTForImageClassification
from transformers import AutoImageProcessor, AutoModelForImageClassification
from PIL import Image, UnidentifiedImageError
import torch
import json
import random


cat_dir = Path("Cat_250")
dog_dir = Path("Dog_250")

cat_images = sorted(p for p in cat_dir.iterdir() if p.is_file())
dog_images = sorted(p for p in dog_dir.iterdir() if p.is_file())

if len(cat_images) != 250 or len(dog_images) != 250:
    raise ValueError(
        f"Expected 250 cat and 250 dog images, "
        f"found {len(cat_images)} cats and {len(dog_images)} dogs."
    )

samples = (
    [(image, 1) for image in cat_images]
    + [(image, 0) for image in dog_images]
)

random.seed(42)
random.shuffle(samples)


# processor = AutoImageProcessor.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_dog_vs_cat_image_classification")
# model =  AutoModelForImageClassification.from_pretrained("DunnBC22/vit-base-patch16-224-in21k_dog_vs_cat_image_classification", device_map="auto")
processor = AutoImageProcessor.from_pretrained("google/vit-base-patch16-224")
model = AutoModelForImageClassification.from_pretrained("google/vit-base-patch16-224", device_map="auto")
model.eval()

# cat_class = "Cat"
# dog_class = "Dog"
# cat_idx = int(model.config.label2id[cat_class])
# dog_idx = int(model.config.label2id[dog_class])

num_layers = len(model.vit.layers)


file = "data/cat_dog/vit_prob.jsonl"
output_path = Path(file)
output_path.parent.mkdir(parents=True, exist_ok=True)


import json

with open("imagenet_class_index.json") as f:
    classes = json.load(f)

cat_keywords = [
    "cat", "cougar", "lynx", "leopard",
    "jaguar", "lion", "tiger", "cheetah"
]

exclude_keywords = [
    "beetle", "polecat", "fish", "oyster", "catamaran", "sea"
]

feline_classes = []

for idx, (wnid, name) in classes.items():
    name_lower = name.lower()

    if (
        any(k in name_lower for k in cat_keywords)
        and not any(k in name_lower for k in exclude_keywords)
    ):
        feline_classes.append((int(idx), wnid, name))

for idx, wnid, name in feline_classes:
    print(idx, wnid, name)

feline_indices = [feline[0] for feline in feline_classes]

dog_keywords = [
    "dog", "terrier", "spaniel", "retriever",
    "shepherd", "hound", "poodle", "collie",
    "bulldog", "mastiff", "pinscher", "schnauzer",
    "chihuahua", "corgi", "dachshund", "beagle",
    "boxer", "husky", "malamute", "rottweiler",
    "doberman", "pug", "pomeranian", "samoyed",
    "greyhound", "whippet", "wolfhound", "setter",
    "pointer", "bloodhound", "basset", "pekinese",
    "pekingese", "shih-tzu", "papillon", "affenpinscher",
    "chow", "basenji", "borzoi", "briard",
    "komondor", "kuvasz", "vizsla", "weimaraner",
    "schipperke", "kelpie", "dhole"
]

exclude_keywords = [
    "beetle", "polecat", "fish", "oyster",
    "catamaran", "sea", "hotdog"
]

dog_classes = []

for idx, (wnid, name) in classes.items():
    name_lower = name.lower()

    if (
        any(k in name_lower for k in dog_keywords)
        and not any(k in name_lower for k in exclude_keywords)
    ):
        dog_classes.append((int(idx), wnid, name))

for idx, wnid, name in dog_classes:
    print(idx, wnid, name)

dog_indices = [dog[0] for dog in dog_classes]


with output_path.open("w", encoding="utf-8") as f:

    for image_number, (image_path, label) in enumerate(samples, start=1):

        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, UnidentifiedImageError):
            print(f"Skipping invalid image: {image_path}")
            continue

        inputs = processor(images=image, return_tensors="pt")
        outputs = model(**inputs, output_hidden_states=True)

        logits = outputs.logits

        # model predicts one of the 1000 ImageNet classes
        predicted_class_idx = logits.argmax(-1).item()
        predicted_class = model.config.id2label[predicted_class_idx]
        print("Predicted class:", predicted_class)

        num_tokens = outputs.hidden_states[0].shape[1]

        initial_strengths = torch.zeros(num_tokens, len(model.vit.layers))

        # Note: This does not include dropout layer, only ViT layers
        for token in range(num_tokens):
            for layer in range(len(model.vit.layers)):

                # Number of tokens is 197 (CLS + number of patches)
                token_activations = outputs.hidden_states[layer][0][token]

                # Apply unembedding matrix
                interpreted_activations = model.classifier(token_activations)
                interpreted_softmaxes = torch.softmax(interpreted_activations, dim=0)

                cat_probs = interpreted_softmaxes[feline_indices].sum()
                dog_probs = interpreted_softmaxes[dog_indices].sum()

                cat_raw = cat_probs.unsqueeze(0)
                dog_raw = dog_probs.unsqueeze(0)

                cat_dog_raw = torch.cat((cat_raw, dog_raw))
                initial_strengths[token][layer] = cat_dog_raw[0] - cat_dog_raw[1]

            print(f"Finished token: {token}")

        data = {
            "label": label,
            "result": {
                "query": {
                    "query": str(image_path),
                    "query_probs": initial_strengths.detach().cpu().tolist(),
                    "token_score": 0
                },
                "answer": {
                    "answer": predicted_class,
                    "tokenized_answer": predicted_class
                }
            }
        }

        f.write(json.dumps(data) + "\n")

        print(
            f"Finished image {image_number}/{len(samples)}: {image_path}"
        )

print(f"Completed writing initial strengths to {file}.")
