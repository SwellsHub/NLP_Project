import torch
from torchtext.legacy import data
from torchtext.legacy import datasets

from CNN import CNN
import torch.nn as nn
import torch.nn.functional as F
import random

model = TheModelClass()
model = load_state_dict(torch.load('tut5-model.pt'))

test_loss, test_acc = evaluate(model, test_iterator, criterion)

print(f'Test Loss: {test_loss:.3f} | Test Acc: {test_acc*100:.2f}%')

import spacy
nlp = spacy.load('en_core_web_sm')

def predict_class(model, sentence, min_len = 4):
    model.eval()
    tokenized = [tok.text for tok in nlp.tokenizer(sentence)]
    if len(tokenized) < min_len:
        tokenized += ['<pad>'] * (min_len - len(tokenized))
    indexed = [TEXT.vocab.stoi[t] for t in tokenized]
    tensor = torch.LongTensor(indexed).to(device)
    tensor = tensor.unsqueeze(1)
    preds = model(tensor)
    max_preds = preds.argmax(dim = 1)
    return max_preds.item()

pred_class = predict_class(model, "I am very angry")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

pred_class = predict_class(model, "I don't know if I'm going to make it")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

pred_class = predict_class(model, "You better give me a lower price!")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')

pred_class = predict_class(model, "Thank you so much!")
print(f'Predicted class is: {pred_class} = {LABEL.vocab.itos[pred_class]}')