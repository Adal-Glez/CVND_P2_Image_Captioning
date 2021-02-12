import torch
import torch.nn as nn
import torchvision.models as models


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
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers=num_layers, dropout = 0.2, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        
        """
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(input_size = embed_size, 
                            hidden_size = hidden_size, 
                            num_layers=num_layers,
                            dropout = 0.2,
                            batch_first = True)
        self.fc = nn.Linear(hidden_size, vocab_size)
        """
    def forward(self, features, captions):
        
        batch_size = features.shape[0]
        captions_trimmed = captions[..., :-1]
        embed = self.embed(captions_trimmed)
        inputs = torch.cat([features.unsqueeze(1), embed], 1)
        lstm_out, self.hidden = self.lstm(inputs)
        outputs = self.fc(lstm_out)
        return outputs
        """
        captions = captions[..., :-1]
        captions = self.embed(captions)
        inputs = torch.cat([features.unsqueeze(1), embed], dim = 1)
        lstm_out, _ = self.lstm(inputs)
        outputs = self.fc(lstm_out)
        """
        
    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        tokens = []
        for i in range(max_len):
            # Step through the sequence one element at a time.
            # after each step, hidden contains the hidden state.
            #out, hidden = lstm(i.view(1, 1, -1), hidden)
            
            lstm_output, states = self.lstm(inputs, states)
            out = self.fc(lstm_output)
            
            argmax = torch.argmax(out,dim=2)
            ind = argmax.item()
            tokens.append(argmax.item())
            
            #inputs = self.embed(argmax[1].long()).unsqueeze(1)
            if ind == 1:  # <end>
                break
            inputs = self.embed(argmax)
        return tokens
            
            

            
            