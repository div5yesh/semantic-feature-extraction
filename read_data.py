#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 15:30:58 2019

@author: Binit Gajera
"""
#%%
import matplotlib.pyplot as plt
import numpy as np
import logging
import torch
from pytorch_pretrained_bert import BertTokenizer, BertModel, BertForMaskedLM

#%matplotlib inline

# %%
class Embeddings:

    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertModel.from_pretrained('bert-base-uncased')
        self.model.eval()

    def getEmbeddings(self, word):
        # Load pre-trained model tokenizer (vocabulary)

        # Define a new example sentence with multiple meanings of the word "bank"
        # text = word

        # Add the special tokens.
        # marked_text = "[CLS] " + text + " [SEP]"
        # marked_text = text

        # Split the sentence into tokens.
        tokenized_text = self.tokenizer.tokenize(word)
        # print(tokenized_text)

        # Map the token strings to their vocabulary indeces.
        indexed_tokens = self.tokenizer.convert_tokens_to_ids(tokenized_text)

        # Display the words with their indeces.
        # for tup in zip(tokenized_text, indexed_tokens):
        #     print('{:<12} {:>6,}'.format(tup[0], tup[1]))

        # Mark each of the 22 tokens as belonging to sentence "1".
        # segments_ids = [1] * len(tokenized_text)

        # print (segments_ids)

        # Convert inputs to PyTorch tensors
        tokens_tensor = torch.tensor([indexed_tokens])
        # segments_tensors = torch.tensor([segments_ids])

        # Load pre-trained model (weights)

        # Put the model in "evaluation" mode, meaning feed-forward operation.

        # Predict hidden states features for each layer
        with torch.no_grad():
            encoded_layers, _ = self.model(tokens_tensor)
            # print(len(encoded_layers))

        # print ("Number of layers:", len(encoded_layers))
        # layer_i = 0

        # print ("Number of batches:", len(encoded_layers[layer_i]))
        # batch_i = 0

        # print ("Number of tokens:", len(encoded_layers[layer_i][batch_i]))
        # token_i = 0

        # print ("Number of hidden units:", len(encoded_layers[layer_i][batch_i][token_i]))

        # For the 5th token in our sentence, select its feature values from layer 5.
        # token_i = 0
        # layer_i = 0
        # vec = encoded_layers[layer_i][batch_i][token_i]

        # # Plot the values as a histogram to show their distribution.
        # plt.figure(figsize=(10,10))
        # plt.hist(vec, bins=200)
        # plt.show()

        # `encoded_layers` is a Python list.
        # print('     Type of encoded_layers: ', type(encoded_layers))

        # # Each layer in the list is a torch tensor.
        # print('Tensor shape for each layer: ', encoded_layers[0].size())

        # Concatenate the tensors for all layers. We use `stack` here to
        # create a new dimension in the tensor.
        token_embeddings = torch.stack(encoded_layers, dim=0)

        token_embeddings.size()

        # Remove dimension 1, the "batches".
        token_embeddings = torch.squeeze(token_embeddings, dim=1)

        token_embeddings.size()

        # Swap dimensions 0 and 1.
        token_embeddings = token_embeddings.permute(1,0,2)

        token_embeddings.size()

        return token_embeddings[0].reshape(-1)

# %%
