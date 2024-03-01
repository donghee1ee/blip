import os
import json
from PIL import Image

from transformers.utils import logging
import torch
from datasets import Dataset
import pandas as pd

logger = logging.get_logger(__name__)

def test_snapme(processor, trainer):
    # TODO optimize. prompt..
    with open('/nfs_share2/shared/from_donghee/snapme/snapme_processed.json', 'r') as f:
        labels = json.load(f)

    id2gt = {entry['id']: entry['gt_int'] for entry in labels}
    id2gt_string = {entry['id']: entry['gt'] for entry in labels} # store ingredients in string [milk, sugar, flour...]

    samples = []
    base_path = '/nfs_share2/shared/from_donghee/snapme/snapme_mydb'

    for filename in os.listdir(base_path):
        file_path = os.path.join(base_path, filename)
        if filename.split('/')[-1] in id2gt: 
            try:
                with Image.open(file_path) as img:
                    samples.append({
                        'image_path': file_path,
                        'labels': id2gt[filename.split('/')[-1]]
                    })
            except Exception as e:
                print(f"Error processing {file_path}: {e}")

    def process_snapme(examples):
        prompt = 'Question: What are the ingredients I need to make this food? Answer:'

        images = []
        for path in examples['image_path']:
            try:
                img = Image.open(path)
                images.append(img)
            except:
                print("error")
                break

        inputs = processor(images=images, text=[prompt]*len(images), return_tensors='pt')

        labels = []
        for label in examples['labels']:
            entry = torch.tensor([0]*1488, dtype=torch.float32) ## TODO float32?
            idx = torch.tensor(label).nonzero(as_tuple=True)[0]
            entry[idx] = 1
            labels.append(entry)
        
        inputs['labels'] = torch.stack(labels)

        return inputs

    dataset = Dataset.from_pandas(pd.DataFrame(samples))

    snapme_dataset = dataset.map(process_snapme, batched=True)

    # or trainer.predict
    test_result = trainer.evaluate(snapme_dataset)
    logger.info(test_result)