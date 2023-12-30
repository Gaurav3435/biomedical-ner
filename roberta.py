
# visualization libraries
import matplotlib.pyplot as plt
import numpy as np

# pytorch libraries
import torch # the main pytorch library
import torch.nn as nn # the sub-library containing Softmax, Module and other useful functions
import torch.optim as optim # the sub-library containing the common optimizers (SGD, Adam, etc.)

# huggingface's transformers library
from transformers import RobertaForTokenClassification, RobertaTokenizer

# huggingface's datasets library
from datasets import load_dataset


import os
print(os.getcwd())
roberta_version = 'roberta-base'
tokenizer = RobertaTokenizer.from_pretrained(roberta_version)

dataset = load_dataset("ncbi_disease")

print(dataset)

num_labels = dataset['train'].features['ner_tags'].feature.num_classes
ner_feature = dataset['train'].features['ner_tags']
label_names = ner_feature.feature.names
print('label_names', label_names)

encodings = tokenizer(dataset['train'][0]['tokens'], truncation=True, padding='max_length', is_split_into_words=True)
print(encodings)

labels = dataset['train'][0]['ner_tags'] + [0] * (tokenizer.model_max_length - len(dataset['train'][0]['ner_tags']))
print(labels)

def add_encodings(example):
    """Processing the example
    
    Args:
        example (dict): The dataset example.
    
    Returns:
        dict: The dictionary containing the following updates:
            - input_ids: The list of input ids of the tokens.
            - attention_mask: The attention mask list.
            - ner_tags: The updated ner_tags.
    
    """
    if len(example['tokens'])!=0:
      # get the encodings of the tokens. The tokens are already split, that is why we must add is_split_into_words=True
      encodings = tokenizer(example['tokens'], truncation=True, padding='max_length', is_split_into_words=True)
      # extend the ner_tags so that it matches the max_length of the input_ids
      labels = example['ner_tags'] + [0] * (tokenizer.model_max_length - len(example['ner_tags']))
      # return the encodings and the extended ner_tags
      return { **encodings, 'labels': labels }
    else:
      # get the encodings of the tokens. The tokens are already split, that is why we must add is_split_into_words=True
      encodings = tokenizer(['.'], truncation=True, padding='max_length', is_split_into_words=True)
      # extend the ner_tags so that it matches the max_length of the input_ids:
      labels =  [0] * (tokenizer.model_max_length - len(example['ner_tags']))
      # return the encodings and the extended ner_tags
      return { **encodings, 'labels': labels }

dataset = dataset.map(add_encodings)

dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

labels = dataset['train'].features['ner_tags'].feature
label2id = { k: labels.str2int(k) for k in labels.names }
id2label = { v: k for k, v in label2id.items() }

print('ID to label',id2label)
print('label to ID',label2id)

# initialize the model and provide the 'num_labels' used to create the classification layer
model = RobertaForTokenClassification.from_pretrained(roberta_version, num_labels=num_labels)
# assign the 'id2label' and 'label2id' model configs
model.config.id2label = id2label
model.config.label2id = label2id

import evaluate
metric = evaluate.load('seqeval')

import numpy as np

def compute_metrics(eval_preds): 
    logits, labels = eval_preds
    predictions = np.argmax(logits, axis=-1)
    true_labels = [[label_names[l] for l in label if l!=-100] for label in labels]
    true_predictions = [[label_names[p] for p,l in zip(prediction, label) if l!=-100]
                        for prediction, label in zip(predictions, labels)]
    all_metrics = metric.compute(predictions=true_predictions, references=true_labels)
    return {"precision": all_metrics['overall_precision'],
            "recall": all_metrics['overall_recall'],
            "f1": all_metrics['overall_f1'],
            "accuracy": all_metrics['overall_accuracy']}

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
batch = data_collator([dataset['train'][i] for i in range(2)])



from transformers import TrainingArguments

args = TrainingArguments("roberta-base",
                         evaluation_strategy = "epoch",
                         save_strategy="epoch",
                         learning_rate = 1e-5,
                         num_train_epochs=15,
                         weight_decay=0.01)

from transformers import Trainer
trainer = Trainer(model=model,
                  args=args,
                  train_dataset = dataset['train'],
                  eval_dataset = dataset['validation'],
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer)
print('training started')
trainer.train()
print('training ended')

from transformers import pipeline

checkpoint = "robert-base/checkpoint-2040"
token_classifier = pipeline(
    "token-classification", model=checkpoint, aggregation_strategy="simple"
)
print("Sentence:",' '.join(dataset['train'][0]['tokens']))
print('Actual labels',dataset['train'][0]['ner_tags_str'])
token_classifier(' '.join(dataset['train'][0]['tokens']))
