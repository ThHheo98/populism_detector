import torch
from transformers import get_linear_schedule_with_warmup, AdamW, AutoModel, AutoTokenizer, \
    AutoModelForSequenceClassification
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, Dataset
import datetime
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False, normalization=True)


def encode_for_bert(df, tokenizer, column_name):
    input_ids = []
    attention_masks = []
    for sent in df[[column_name]].values:
        sent = sent.item()
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=65,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    inputs = {
        'input_word_ids': input_ids,
        'input_mask': attention_masks
    }
    return inputs


def make_train_dataloader(df, column_name, tokenizer, batch_size=8):
    input_ids, attention_masks = encode_for_bert(df, tokenizer, column_name).values()
    labels = torch.tensor(df.label.astype(int).values)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    return DataLoader(dataset, sampler=RandomSampler(dataset), batch_size=batch_size)


def make_test_dataloader(df, column_name, tokenizer, batch_size=8):
    input_ids, attention_masks = encode_for_bert(df, tokenizer, column_name).values()
    dataset = TensorDataset(input_ids, attention_masks)
    return DataLoader(dataset, sampler=SequentialSampler(dataset), batch_size=batch_size)


def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)


def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
