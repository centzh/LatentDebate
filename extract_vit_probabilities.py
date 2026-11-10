from datasets import load_dataset
import logging
from transformers import AutoImageProcessor, AutoModelForImageClassification
import numpy as np
import json
import torch
from tqdm import tqdm
from IPython import embed

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_label(dataset, label_name: str):
    """Get the label index for a given class label"""
    return dataset.features.get('label').str2int(label_name)

def get_label_name(dataset, label_idx: int):
    """Get the class label for a given label index"""
    return dataset.features.get('label').int2str(label_idx)

def get_class_instances(dataset, class_label: str):
    """Get all instances of a given class label from a dataset"""
    class_label = get_label(dataset, class_label)
    labels = np.asarray(dataset['label'])
    indices = np.nonzero(labels == class_label)[0]
    return dataset.select(indices)

def main():

    test_dataset = load_dataset("uoft-cs/cifar10", split='test')
    logger.info(f"Loaded CIFAR-10 test dataset with {len(test_dataset)} instances")

    class_labels = test_dataset.features.get('label').names
    logger.info(f"CIFAR-10 class labels: {class_labels}")

    test_cats = get_class_instances(test_dataset, 'cat')
    logger.info(f"Loaded {len(test_cats)} cat instances")

    test_dogs = get_class_instances(test_dataset, 'dog')
    logger.info(f"Loaded {len(test_dogs)} dog instances")

    cat_label = get_label(test_cats, 'cat')

    processor = AutoImageProcessor.from_pretrained("nateraw/vit-base-patch16-224-cifar10")
    model = AutoModelForImageClassification.from_pretrained("nateraw/vit-base-patch16-224-cifar10", device_map="auto")
    model.eval()
    num_layers = len(model.vit.layers)

    with open('test_cats.jsonl', 'w') as f:
        datasets = [test_cats, test_dogs]
        for dataset in datasets:
            for image_number, instance in enumerate(tqdm(dataset, desc="Calculating initial strengths")):
                    image_path = instance.get('img', None)
                    label = instance.get('label', None)
                    logger.info(f"Processing image {image_number}/{len(dataset)}")

                    inputs = processor(images=image_path, return_tensors="pt")
                    outputs = model(**inputs, output_hidden_states=True)

                    logits = outputs.logits
                    pred_label = logits.argmax(-1).item()
                    pred_label_name = get_label_name(dataset, pred_label)

                    num_tokens = outputs.hidden_states[0].shape[1]
                    initial_strengths = torch.zeros(num_tokens, num_layers)

                    hidden_states = [layer.detach().numpy() for layer in outputs.hidden_states]
                    hidden_states = np.asarray(hidden_states)
                    hidden_states = np.squeeze(hidden_states)
                    hidden_states = torch.from_numpy(hidden_states)

                    interpreted_activations = model.classifier(hidden_states)
                    interpreted_softmaxes = torch.softmax(interpreted_activations, dim=-1)
                    cat_probs = interpreted_softmaxes[:, :, cat_label]
                    initial_strengths = cat_probs

                    data = {
                        "label": label,
                        "result": {
                            "query": {
                                "query": str(image_path),
                                "query_probs": initial_strengths.tolist(),
                            },
                            "answer": {
                                "answer": pred_label_name,
                            }
                        }
                    }
                  
                    f.write(json.dumps(data) + "\n")

        logger.info(f"Completed writing initial strengths to file.")

if __name__ == "__main__":
    main()
