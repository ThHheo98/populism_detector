import torch
from transformers import get_linear_schedule_with_warmup,AdamW,AutoModel, AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import TensorDataset,DataLoader, RandomSampler, SequentialSampler, Dataset
import datetime
import numpy as np

tokenizer = AutoTokenizer.from_pretrained("vinai/bertweet-base", use_fast=False, normalization=True)

def bert_encode(df, tokenizer, column_name):
    input_ids = []
    attention_masks = []
    for sent in df[[column_name]].values:
        sent = sent.item()
        encoded_dict = tokenizer.encode_plus(
            sent,
            add_special_tokens=True,
            max_length=65,
            pad_to_max_length=True,
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


def make_dataloader(df, column_name, batch_size=8):
    encoded = bert_encode(df, tokenizer, column_name)
    encoded_labels = df.label.astype(int)

    input_ids, attention_masks = encoded.values()

    print(input_ids.shape)
    print(attention_masks.shape)

    labels = torch.tensor(encoded_labels.values)
    dataset = TensorDataset(input_ids, attention_masks, labels)
    dataloader = DataLoader(
        dataset,
        sampler=RandomSampler(dataset),
        batch_size=batch_size
    )
    return dataloader

def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def format_time(elapsed):
    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))
