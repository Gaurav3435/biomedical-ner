# Biomedical Named Entity Recognition (NER)

This repository contains fine-tuned models for Named Entity Recognition (NER) in biomedical texts. The models are fine-tuned on the NCBI disease dataset using several pre-trained transformer-based architectures including:

- **RoBERTa**
- **XLM-RoBERTa**
- **GPT-2**
- **BioBERT**
- **DistilBERT**


### Models and Techniques

The following models are employed:
- **RoBERTa**: Robustly optimized BERT pretraining approach.
- **XLM-RoBERTa**: A multilingual variant of RoBERTa for cross-lingual tasks.
- **GPT-2**: A generative model fine-tuned for NER.
- **BioBERT**: A BERT-based model pre-trained on large-scale biomedical corpora.
- **DistilBERT**: A smaller, faster variant of BERT with competitive performance.

## Overview

This repository contains implementations for Named Entity Recognition (NER) in biomedical text, leveraging various state-of-the-art transformer models. The focus is on disease entity recognition, a critical task in biomedical text mining that assists in extracting relevant medical information from scientific literature. This work extends previous research by fine-tuning a selection of pre-trained models, including BioBERT, RoBERTa, and XLM-RoBERTa, to improve disease recognition performance in biomedical abstracts.

The project is part of research focused on improving biomedical information extraction systems through fine-tuning pre-trained transformer models.

## Introduction

Biomedical Named Entity Recognition (NER) plays a crucial role in identifying relevant biomedical entities like diseases, genes, chemicals, and proteins from biomedical texts. In this repository, we fine-tune a set of pre-trained transformer models on the NCBI disease dataset to enhance disease entity recognition.

The research paper presents a comparative analysis of various models for biomedical NER, demonstrating their effectiveness and limitations when applied to disease identification in biomedical abstracts.

## Models and Techniques

We utilized several transformer-based models, each offering unique advantages in terms of model size, speed, and language versatility:

- **BioBERT**: A domain-specific variation of BERT pre-trained on large biomedical corpora, designed to enhance biomedical entity recognition tasks.
- **RoBERTa**: A robustly optimized variant of BERT, fine-tuned for disease recognition tasks in biomedical texts.
- **XLM-RoBERTa**: A multilingual version of RoBERTa for cross-lingual NER, capable of handling biomedical texts in multiple languages.
- **GPT-2**: A generative pre-trained model, repurposed for sequence labeling tasks such as NER.
- **DistilBERT**: A smaller and faster variant of BERT, offering a trade-off between performance and computational efficiency.

These models are fine-tuned specifically for disease entity recognition, making them effective for extracting biomedical information from scientific abstracts.

## Dataset

The models are trained and evaluated on the **NCBI Disease Dataset**, a widely-used benchmark dataset in biomedical NER. The dataset consists of abstracts from PubMed and contains labeled disease entities, allowing for evaluation of model performance on disease identification.

### NCBI Disease Dataset
- **Source**: The dataset is available from the NCBI website.
- **Task**: Disease recognition within biomedical text, particularly focusing on disease entities in abstract sentences.


### Usage

Clone this repository and follow the instructions below to fine-tune and run the models on biomedical texts.

```bash
git clone https://github.com/Gaurav3435/biomedical-ner.git
cd biomedical-ner
# Instructions to install dependencies and run the fine-tuned models.
```

## Dependencies

To use the code and models, you'll need the following dependencies:

- Python 3.x
- PyTorch
- Hugging Face Transformers
- scikit-learn
- tqdm
- pandas


### Results
The models demonstrate strong performance in identifying biomedical entities, particularly in the domain of disease recognition. Evaluate them on your own datasets to explore their effectiveness further.
