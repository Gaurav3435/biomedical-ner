import pandas as pd
from datasets import load_dataset
from transformers import GPT2LMHeadModel, GPT2TokenizerFast
from transformers import AutoTokenizer

data = load_dataset('dataset/conll2003')

tags = data['train'].features['ner_tags'].feature

index2tag = {idx:tag for idx, tag in enumerate(tags.names)}

tag2index = {tag:idx for idx, tag in enumerate(tags.names)}

def create_tag_names(batch):
    tag_name = {'ner_tags_str': [tags.int2str(idx) for idx in batch['ner_tags']]}
    return tag_name
data = data.map(create_tag_names)

print("Keys of data:",data['train'][:].keys())

model_checkpoint = "distilbert-base-cased"
tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)
tokenizer.add_special_tokens({'pad_token': '[PAD]'})

print("Check is the token is fast:",tokenizer.is_fast)

inputs = data['train'][0]['tokens']

print('Checking inputs',inputs)

inputs = tokenizer(inputs, is_split_into_words=True)

print('Check the tokenized input:',inputs.tokens())

print('Check the tokenst:',data['train'][0]['tokens'])
print('Check the ner_tags_str:',data['train'][0]['ner_tags_str'])

def align_labels_with_tokens(labels, word_ids):
    new_labels = []
    current_word=None
    for word_id in word_ids:
        if word_id != current_word:
            current_word = word_id
            label = -100 if word_id is None else labels[word_id]
            new_labels.append(label)
        elif word_id is None:
            new_labels.append(-100)

        else:
            label = labels[word_id]
            if label%2==1:
                label = label + 1
            new_labels.append(label)
    
    return new_labels

labels = data['train'][0]['ner_tags']
word_ids = inputs.word_ids()

align_labels_with_tokens(labels, word_ids)

def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(examples['tokens'], truncation=True, is_split_into_words=True)
    all_labels = examples['ner_tags']
    new_labels = []
    for i, labels in enumerate(all_labels):
        word_ids = tokenized_inputs.word_ids(i)
        new_labels.append(align_labels_with_tokens(labels, word_ids))

    tokenized_inputs['labels'] = new_labels

    return tokenized_inputs

tokenized_datasets = data.map(tokenize_and_align_labels, batched=True, remove_columns=data['train'].column_names)
print('Check the tokenized_datasets:',tokenized_datasets)

print('input_ids',tokenized_datasets['train'][:]['input_ids'][0])
print('attention_mask',tokenized_datasets['train'][:]['attention_mask'][0])
print('labels',tokenized_datasets['train'][:]['labels'][0])

from transformers import DataCollatorForTokenClassification

data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
batch = data_collator([tokenized_datasets['train'][i] for i in range(2)])

import evaluate
metric = evaluate.load('seqeval')
ner_feature = data['train'].features['ner_tags']
label_names = ner_feature.feature.names

labels = data['train'][0]['ner_tags']
labels = [label_names[i] for i in labels]
predictions = labels.copy()
predictions[2] = "O"

metric.compute(predictions=[predictions], references=[labels])

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

id2label = {i:label for i, label in enumerate(label_names)}
label2id = {label:i for i, label in enumerate(label_names)}

from transformers import AutoModelForTokenClassification

model = AutoModelForTokenClassification.from_pretrained(
                                                    'gpt2',
                                                    id2label=id2label,
                                                    label2id=label2id)

print('Numeber of labes:',model.config.num_labels)

from transformers import TrainingArguments

args = TrainingArguments("gpt2",
                         evaluation_strategy = "epoch",
                         save_strategy="epoch",
                         learning_rate = 2e-5,
                         num_train_epochs=3,
                         weight_decay=0.01)
from transformers import Trainer
trainer = Trainer(model=model,
                  args=args,
                  train_dataset = tokenized_datasets['train'],
                  eval_dataset = tokenized_datasets['validation'],
                  data_collator=data_collator,
                  compute_metrics=compute_metrics,
                  tokenizer=tokenizer)

trainer.train()

from transformers import pipeline

checkpoint = "gpt2/checkpoint-5268"
token_classifier = pipeline(
    "token-classification", model=checkpoint, aggregation_strategy="simple"
)

print(token_classifier("My name is Laxmi Kant Tiwari. I work at KGP Talkie and live in Mumbai"))
