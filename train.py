import json
import logging
import os
import argparse

import torch
from torch.nn.parallel import DistributedDataParallel as DDP
from transformers import InstructBlipProcessor, TrainingArguments, Trainer

from data.dataset import load_datasets, CustomDataCollator, collator, Recipe1M_Collator, load_datasets_for_distributed
from data.utils import Vocabulary
from model.modeling_instructblip import FreezeInstructBlipForConditionalGeneration
from common.dist_utils import init_distributed_mode
from common.logger import setup_logger

# TODO
# 2. compute_metric : F1/IoU
# 3. BERT
# 4. Optimizing dataset preprocess - on the fly
# 5. log only main process

def pretty_print(args):
    args_dict = vars(args)
    formatted_args = json.dumps(args_dict, indent=4, sort_keys=True)
    logging.info("Args: \n"+formatted_args)

def parse_args():
    parser = argparse.ArgumentParser(description="Training script for distributed InstructBlip.")

    parser.add_argument('--project_name', type=str, default='temp')
    # /path/to/Recipe1M/dataset
    parser.add_argument('--dataset_path', type=str, default='/nfs_share2/code/donghee/inversecooking/data', help='path containing Recipe1M dataset')

    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--training_samples', type=int, default=1000, help='number of training sample. set to -1 for training on entire dataset')
    parser.add_argument('--eval_samples', type=int, default=100, help='number of eval/test sample. set to -1 for evaluating on entire dataset')
    parser.add_argument('--pre_map', type=bool, default=True, help='process data before forward')

    parser.add_argument(
        '--model_name', 
        type=str, 
        default='Salesforce/instructblip-flan-t5-xl',
        choices=['Salesforce/instructblip-flan-t5-xl', 'Salesforce/instructblip-flan-t5-xxl', 'Salesforce/instructblip-vicuna-7b'],
        help="Specifies the model to use. Choose from 'Salesforce/instructblip-flan-t5-xl' (default), "
            "'Salesforce/instructblip-flan-t5-xxl', or 'Salesforce/instructblip-vicuna-7b'."
    )

    args = parser.parse_args()
    
    args.output_dir= os.path.join("./outputs", args.project_name)
    args.logging_dir = os.path.join('./logs', args.project_name)
    if 't5' in args.model_name:
        args.decoder_only = False
    else:
        args.decoder_only = True

    return args

def train(args):

    model = FreezeInstructBlipForConditionalGeneration.from_pretrained(args.model_name)
    processor = InstructBlipProcessor.from_pretrained(args.model_name)

    datasets = load_datasets( 
        processor=processor, 
        data_dir=args.dataset_path, 
        training_samples=args.training_samples,
        eval_samples=args.eval_samples, 
        pre_map=args.pre_map,
        decoder_only=args.decoder_only
    )
    
    training_args = TrainingArguments(
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        num_train_epochs=args.epochs,
        evaluation_strategy="epoch",
        logging_dir=args.logging_dir,
        logging_strategy = 'steps',
        logging_steps=10,
        do_train=True,
        do_eval=True,
        output_dir=args.output_dir,
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model='loss',
        dataloader_num_workers=4,
        ddp_find_unused_parameters=False,
    )

    # not saving LLM, VIT paramters
    model.set_ignore_keys()
    # data_collator = CustomDataCollator(processor, args.decoder_only) # tokenize on the fly

    # TODO: compute_metrics
    trainer = Trainer( 
        model=model,
        args=training_args,
        train_dataset=datasets['train'],
        eval_dataset=datasets['val'],
        # compute_metrics=compute_metrics,
        # data_collator=data_collator
    )

    # Train the model
    trainer.train()

    # Save
    processor.save_pretrained(os.path.join(args.output_dir, 'best'))
    model.save_pretrained(os.path.join(args.output_dir, 'best'))

    print("* Test start *")
    test_results = trainer.evaluate(datasets['test'])
    print(test_results)


if __name__ == '__main__':
    args = parse_args()
    setup_logger(args)
    pretty_print(args)

    train(args)