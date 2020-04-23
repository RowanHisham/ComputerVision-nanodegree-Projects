import torch
import torch.nn as nn
import torchvision.models as models
import numpy as np 

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(DecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        '''
        Embedding layer that turns each word in captions into a vector of a specified size (embed size)
        '''
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        '''
        The LSTM takes embedded word vectors (of size embed_size) as inputs and outputs hidden states of size hidden_dim
        '''
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,batch_first=True)
       
        '''
        The linear layer that maps the hidden state to output dimension (vocab_size)
        '''
        self.hidden2word = nn.Linear(hidden_size, vocab_size)
              
    
    def forward(self, features, captions):
        '''
        Remove <end> to avoind "AssertionError: The shape of the decoder output is incorrect." in Notebook 1
        Because we don't need a prediction for a word after <end> token
        '''
        captions = captions[:, :-1] 
        
        '''
        Create embedded word vectors for each word in the captions
        # output shape : (batch_size, caption length , embed_size)
        '''
        captions = self.word_embeddings(captions)

        '''
        Concatenate the Image features and the Captions embedding 
        # Features shape : (batch_size, embed_size)
        # Word embeddings shape : (batch_size, caption length , embed_size)
        # output shape : (batch_size, caption length, embed_size)
        '''
        inputs = torch.cat((features.unsqueeze(1), captions), dim=1)
        
        '''
        Get the output and hidden state by passing the lstm over the embeddings
        # output shape : (batch_size, caption length , hidden_size)
        '''
        lstm_out, _ = self.lstm(inputs)

        '''
        Pass the lstm_out through the last FC layer to get the final prediction of each embedded word over
        the vocab_size
        # output shape : (batch_size, caption length , vocab_size)
        '''
        outputs = self.hidden2word(lstm_out)

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        outputs = []
#         states = (torch.randn(1, 1, self.hidden_size).to(inputs.device), 
#                   torch.randn(1, 1, self.hidden_size).to(inputs.device))
        
        for i in range(max_len):
            '''
            Get the output and hidden state by passing the lstm over the embeddings
            '''
            output, states = self.lstm(inputs,states)

            '''
            Pass the lstm_out through the last FC layer to get the final prediction of each embedded word over
            the vocab_size
            '''
            output = self.hidden2word(output.squeeze(dim = 1))
            output = torch.argmax(output)
            if (output == 1):
                break
           
            outputs.append(output.item())
            
            '''
            Update the Input of the LSTM to be the current output
            '''
            inputs = self.word_embeddings(output)
            inputs = inputs[None,None,:]
                   
        return outputs
        