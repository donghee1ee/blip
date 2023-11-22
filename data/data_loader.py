# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import torch
import torchvision.transforms as transforms
import torch.utils.data as data
import os
import pickle
import numpy as np
import nltk
from PIL import Image
from data.utils import Vocabulary
import random
import json
import lmdb
from torch.utils.data import Dataset
from typing import Dict
import pandas as pd
from datasets import Features, Value, Array3D, Sequence
from datasets import Dataset as hf_Dataset
from transformers import DataCollatorWithPadding

class Recipe1MDataset(Dataset):
    """
    This Dataset class is designed to work wirh LMDB files, assuming that the images are pre-stored in a LMDB database.
    It DOES NOT support loading images directly from JPEG files.
    """
    def __init__ (self, processor, data_dir, split, max_num_labels, max_num_samples=-1):
        
        self.split = split

        # load 3 files: lmdb, recipe1m pkl, vocab file
        self.image_file = lmdb.open(os.path.join(data_dir, 'lmdb_'+ split), max_readers=1, readonly=True, lock = False, readahead=False, meminit=False)
        self.dataset = pickle.load(open(os.path.join(data_dir, 'recipe1m_'+split+'.pkl'), 'rb'))
        self.ingrs_vocab = pickle.load(open(os.path.join(data_dir, 'recipe1m_vocab_ingrs.pkl'), 'rb'))
        
        self.processor = processor

        self.ids = [] 
        for i, entry in enumerate(self.dataset):
            if len(entry['images']) == 0:
                continue
            self.ids.append(i)

        self.maxnumims = 5
        self.max_num_labels = max_num_labels
        if max_num_samples != -1: 
            # randomly sample subset
            random.shuffle(self.ids)
            self.ids = self.ids[:max_num_samples]

    
    # def get_instrs_vocab(self):
    #     return self.instrs_vocab

    # def get_instrs_vocab_size(self):
    #     return len(self.instrs_vocab)
    
    def get_ingrs_vocab(self):
        return [min(w, key=len) if not isinstance(w, str) else w for w in
                self.ingrs_vocab.idx2word.values()]  # includes 'pad' ingredient
    
    def get_ingrs_vocab_size(self):
        return len(self.ingrs_vocab)
    
    def __len__(self):
        return len(self.ids)
    
    def to_list(self):
        # Convert the dataset to a list of dictionaries
        data_list = []
        for idx in range(len(self)):
            data_point = self.__getitem__(idx)
            data_list.append(data_point)
        return data_list
    
    def __getitem__(self, index):
        
        current_id = self.ids[index]
        sample = self.dataset[current_id]
        recipe_id = sample['id']
        # captions = sample['tokenized']
        paths = sample['images'][0:self.maxnumims]
        if self.split == 'train':
            idx = np.random.randint(0, len(paths))
        else:
            idx = 0
        path = paths[idx]

        try:
            with self.image_file.begin(write=False) as txn:
                image = txn.get(path.encode())
                image = np.fromstring(image, dtype=np.uint8)
                image = np.reshape(image, (256, 256, 3))
            image = Image.fromarray(image.astype('uint8'), 'RGB')
            # pixel_values = self.processor(images=image)['pixel_values'][0] #### TODO 
        except:
            # print ("Image recipe_id not found in lmdb. Loading jpeg file...")
            # image = Image.open(os.path.join(self.root, path[0], path[1],
            #                                 path[2], path[3], path)).convert('RGB')
            print("Not able to load lmdb file")
            pixel_values = None

        # idx = index

        ingr_gt = sample['ingredients']
        random.shuffle(ingr_gt)
        ingr_gt = ", ".join(ingr_gt)
        ingr_gt = ingr_gt.replace('_',' ')
        # text_output = self.processor(text=ingr_gt)
        # input_ids = text_output.input_ids
        # attention_mask = text_output.attention_mask
        # label_ids = text_output.input_ids ### TODO

        # labels = self.dataset[self.recipe_ids[recipe_idx]]['ingredients']
        title = sample['title']
        title = " ".join(title)

        # ingredient integer
        labels = sample['ingredients']
        ilabels_gt = np.ones(self.max_num_labels) * self.ingrs_vocab('<pad>')
        pos = 0

        true_ingr_recipe_idxs = []

        true_ingr_recipe_idxs = []
        for i in range(len(labels)):
            true_ingr_recipe_idxs.append(self.ingrs_vocab(labels[i]))

        for i in range(self.max_num_labels):
            if i >= len(labels):
                label = '<pad>'
            else:
                label = labels[i]
            label_recipe_idx = self.ingrs_vocab(label)
            if label_recipe_idx not in ilabels_gt:
                ilabels_gt[pos] = label_recipe_idx
                pos += 1

        ilabels_gt[pos] = self.ingrs_vocab('<end>')
        ingr_gt_int = torch.from_numpy(ilabels_gt).long()

        question = 'Question: What are the ingredients I need to make this food? Answer:'
        # text_input = self.processor(text=question)
        # qformer_input = self.qformer_tokenizer(question) ## TODO padding
        # qformer_input_ids = text_input.qformer_input_ids
        # qformer_attention_mask = text_input.qformer_attention_mask
        # input_ids = text_input.input_ids
        # attention_mask = text_input.input_ids

        ### TODO
        # llm_input = self.llm_tokenizer(question) ## TODO padding
        # input_ids = llm_input.input_ids
        # attention_mask = llm_input.attention_mask
        ####
        # return {
        #     'pixel_values': pixel_values,
        #     'qformer_input_ids': qformer_input_ids,
        #     'qformer_attention_mask': qformer_attention_mask,
        #     'input_ids': input_ids,
        #     'attention_mask': attention_mask,
        #     'label_ids': label_ids
        # }

        return {
            'image': image,
            'text_input': question,
            'text_output': ingr_gt,
            'text_input_output': question+ingr_gt,
            'ingredient_int': ingr_gt_int,
            # 'title': title,
            # 'recipe_id': recipe_id,
            # 'img_id': path
        }


def collator(data):
    image_list, input_list, output_list, ingr_list, title_list, recipe_id_list, path_list = [], [], [], [], [], [], []

    for sample in data:
        if sample['image'] is not None:
            image_list.append(sample['image'])
            input_list.append(sample['text_input'])
            output_list.append(sample['text_output'])
            ingr_list.append(sample['ingredient_int'])
            title_list.append(sample['title'])
            recipe_id_list.append(sample['recipe_id'])
            path_list.append(sample['img_id'])
    
    return {
        'image' : torch.stack(image_list, dim=0),
        'text_input': input_list,
        'text_output': output_list,
        'ingredient_int': torch.stack(ingr_list, dim=0),
        'title': title_list,
        'recipe_id': recipe_id_list,
        'img_id': path_list
    }


def collate_fn(data):
    
    # Sort a data list by caption length (descending order).
    # data.sort(key=lambda x: len(x[2]), reverse=True)
    data = [sample for sample in data if sample[0] is not None] ##
    image_input, captions, ingrs_gt, img_id, path, pad_value = zip(*data)

    # Merge images (from tuple of 3D tensor to 4D tensor).

    image_input = torch.stack(image_input, 0)
    ingrs_gt = torch.stack(ingrs_gt, 0)

    # Merge captions (from tuple of 1D tensor to 2D tensor).
    lengths = [len(cap) for cap in captions]
    targets = torch.ones(len(captions), max(lengths)).long()*pad_value[0]

    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = cap[:end]

    return image_input, targets, ingrs_gt, img_id, path


def recipe1m_generator(processor, data_dir, split, max_num_labels, max_num_samples):
    recipe_dataset = Recipe1MDataset(processor, data_dir, split, max_num_labels, max_num_samples)
    for i in range(len(recipe_dataset)):
        item = recipe_dataset[i]
        image_array = np.array(item['image'])  # Convert PIL Image to numpy array
        image_array = image_array.transpose((2, 0, 1))  # Convert HWC to CHW format if needed
        yield {
            'image': image_array,
            'text_input': item['text_input'],
            'text_output': item['text_output'],
            'text_input_output': item['text_input_output'],
            'ingredient_int': item['ingredient_int'].tolist()  # Convert torch tensor to list
        }



def load_datasets(batch_size, processor, data_dir, training_samples=-1, max_num_labels=20, pre_map=False, decoder_only=False) -> Dict[str, hf_Dataset]:
    '''
    :param data_dir: recipe1M dataset directory containing 
        1. lmdb dir (lmdb_train, lmdb_val, lmdb_train)
        2. recipe1m_train,val,test pkl - contains ingredient list, recipe instruction, title etc
        3. recipe1m_vocab_ingrs.pkl
    :param training_samples[Optional]: limiting the number of training samples. All training samples are used by default

    Limits the number of validation and test samples to 1500
    '''
    if not pre_map:
        print('batches of data are processed on-the-fly')

    if training_samples != -1:
        print("Limiting the number of training samples to ", training_samples)
    
    def process_data(example):
        # processor.tokenizer.padding_side = 'right'
        # processor.qformer_tokenizer.padding_side = 'right'
        sample = dict()
        
        prompt = processor(
            images=np.array(example['image']),
            text=example['text_input'],
            return_tensors='pt'
        )

        output = processor(
            text=example['text_output'], 
            padding='max_length',
            max_length=128,
            truncation=True,
            return_tensors='pt') # ingredient list

        sample['pixel_values'] = prompt.pixel_values
        sample['qformer_input_ids'] = prompt.qformer_input_ids
        sample['qformer_attention_mask'] = prompt.qformer_attention_mask
        sample['label_ids'] = output.input_ids

        if decoder_only:
            prompt_plus_output = processor(
                text = example['text_input_output'],
                padding='max_length',
                max_length=128,
                truncation=True,
                return_tensors='pt',
            )

            sample['input_ids'] = prompt_plus_output.input_ids
            sample['attention_mask'] = prompt_plus_output.attention_mask
            del prompt_plus_output
        
        else:
            # for encoder-decoder model like t5
            sample['input_ids'] = prompt.input_ids
            sample['attention_mask'] = prompt.attention_mask
            sample['decoder_input_ids'] = output.input_ids
            sample['decoder_attention_mask'] = output.attention_mask

        del prompt, output
        return sample

    datasets = dict()
    features = Features({
        'image': Array3D(dtype="float32", shape=(3, 256, 256)),  # Assuming image converted to 3D array
        'text_input': Value(dtype='string'),
        'text_output': Value(dtype='string'),
        'text_input_output': Value(dtype='string'),
        'ingredient_int': Sequence(feature=Value(dtype='int64'))
    })

    for split in ['train', 'val', 'test']:
        max_num_samples = 1500 if split in ['val', 'test'] else training_samples

        # Define a lambda function that includes the arguments
        generator_function = lambda split=split, max_num_samples=max_num_samples: recipe1m_generator(
            processor=processor,
            data_dir=data_dir,
            split=split,
            max_num_labels=max_num_labels,
            max_num_samples=max_num_samples,
        )

        gen_dataset = hf_Dataset.from_generator(
            generator = generator_function,
            features = features,
        )
        if pre_map:
            datasets[split] = gen_dataset.map(process_data, batched=True, batch_size=batch_size)
        else:
            datasets[split] = gen_dataset

    del gen_dataset

    return datasets

class Recipe1M_Collator(DataCollatorWithPadding):
    def __init__(self, processor):
        super().__init__(tokenizer=processor.tokenizer, padding=True)
        self.processor = processor

    def __call__(self, batch):
        # Initialize lists to store processed fields
        pixel_values = []
        qformer_input_ids = []
        qformer_attention_mask = []
        input_ids = []
        attention_mask = []
        label_ids = []
        ingredient_int = []

        # Process each item in the batch
        for item in batch:
            # Process the image and text input
            encoding = self.processor(
                images=np.array(item['image']), 
                text=item['text_input'], 
                padding='longest', 
                return_tensors='pt'
            )

            # Process the text output for labels
            labels = self.processor(
                text=item['text_output'], 
                padding='longest', 
                return_tensors='pt'
            ).input_ids

            # Append processed items to their respective lists
            pixel_values.append(encoding.pixel_values.squeeze())
            qformer_input_ids.append(encoding.qformer_input_ids.squeeze())
            qformer_attention_mask.append(encoding.qformer_attention_mask.squeeze())
            input_ids.append(encoding.input_ids.squeeze())
            attention_mask.append(encoding.attention_mask.squeeze())
            label_ids.append(labels.squeeze())
            ingredient_int.append(torch.tensor(item['ingredient_int']).long())

        # Combine the lists into a batch dictionary
        processed_batch = {
            'pixel_values': torch.stack(pixel_values),
            'qformer_input_ids': torch.stack(qformer_input_ids),
            'qformer_attention_mask': torch.stack(qformer_attention_mask),
            'input_ids': torch.stack(input_ids),
            'attention_mask': torch.stack(attention_mask),
            'labels': torch.stack(label_ids),
            'ingredient_int': torch.stack(ingredient_int)
        }

        # Utilize parent class method for padding
        return super().__call__(processed_batch)
