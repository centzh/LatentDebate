from datasets import load_dataset
import logging
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import json
import torch
from IPython import embed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_label(dataset, label_name):
    return dataset.features.get('label').str2int(label_name)

def get_label_name(dataset, label_idx):
    return dataset.features.get('label').int2str(label_idx)

def get_class_instances(dataset, class_label):
    class_label = get_label(dataset, class_label)
    labels = np.asarray(dataset['label'])
    indices = np.nonzero(labels == class_label)[0]
    return dataset.select(indices)

test_dataset = load_dataset("uoft-cs/cifar10", split='test')
logger.info(f"Loaded CIFAR-10 test dataset with {len(test_dataset)} instances")

class_labels = test_dataset.features.get('label').names
logger.info(f"CIFAR-10 class labels: {class_labels}")

test_cats = get_class_instances(test_dataset, 'cat')
logger.info(f"Loaded {len(test_cats)} cat instances")

test_dogs = get_class_instances(test_dataset, 'dog')
logger.info(f"Loaded {len(test_dogs)} dog instances")

processor = AutoImageProcessor.from_pretrained("nateraw/vit-base-patch16-224-cifar10")
model = AutoModelForImageClassification.from_pretrained("nateraw/vit-base-patch16-224-cifar10", device_map="auto")
model.eval()
num_layers = len(model.vit.layers)

with open('test_cats.jsonl', 'w') as f:
    datasets = [test_cats, test_dogs]
    for dataset in datasets:
        for image_number, instance in enumerate(dataset):
            image_path = instance.get('img', None)
            label = instance.get('label', None)
            logger.info(f"Processing image: {image_path} with label: {label}")

            inputs = processor(images=image_path, return_tensors="pt")
            outputs = model(**inputs, output_hidden_states=True)
            logger.info(f"Finished processing image: {image_path}")

            logits = outputs.logits
            logger.info(f"Logits: {logits}")
            predicted_class = logits.argmax(-1).item()
            predicted_class_name = get_label_name(dataset, predicted_class)
            logger.info(f"Predicted class: {predicted_class_name}")

            num_tokens = outputs.hidden_states[0].shape[1]
            initial_strengths = torch.zeros(num_tokens, num_layers)
        
            # Note: This does not include dropout layer, only ViT layers
            for token in range(num_tokens):
                for layer in range(len(model.vit.layers)):

                    # Number of tokens is 197 (CLS + number of patches)
                    token_activations = outputs.hidden_states[layer][0][token]

                    # Apply unembedding matrix
                    interpreted_activations = model.classifier(token_activations)
                    interpreted_softmaxes = torch.softmax(interpreted_activations, dim=0)

                    cat_probs = interpreted_softmaxes[get_label(dataset, 'cat')].sum()
                    dog_probs = interpreted_softmaxes[get_label(dataset, 'dog')].sum()

                    cat_raw = cat_probs.unsqueeze(0)
                    dog_raw = dog_probs.unsqueeze(0)

                    cat_dog_raw = torch.cat((cat_raw, dog_raw))
                    initial_strengths[token][layer] = cat_dog_raw[0] - cat_dog_raw[1]

                logger.info(f"Finished token: {token}")

        data = {
            "label": label,
            "result": {
                "query": {
                    "query": str(image_path),
                    "query_probs": initial_strengths.detach().cpu().tolist(),
                    "token_score": 0
                },
                "answer": {
                    "answer": predicted_class_name,
                }
            }
        }

        f.write(json.dumps(data) + "\n")

        logger.info(
            f"Finished image {image_number}/{len(dataset)}: {image_path}"
        )

logger.info(f"Completed writing initial strengths to file.")

            


