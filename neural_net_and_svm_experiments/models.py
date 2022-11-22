import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.layer1 = nn.Linear(input_size, hidden_size, bias=True) 
        self.layer2 = nn.Linear(hidden_size, output_size, bias=True)  
        
    def forward(self, x):
        x = self.layer1(x) 
        x = F.relu(x) 
        x = self.layer2(x)
        return x  
    

class FeatureExtractor(nn.Module):
    
    def __init__(self, seq_length,
                       num_words,
                       hidden_size, 
                       emb_size):
        super().__init__()
        self.layer = MLP(num_words, hidden_size, emb_size)
        self.normalization = nn.LayerNorm(emb_size)
        self.seq_length = seq_length
        self.emb_size = emb_size
    
    def forward(self, x):
        # (bsz, seq_length, num_words) --> (bsz, seq_length, emb_size)
        x = self.layer(x) 
        # Layer norm
        x = self.normalization(x)
        #(bsz, seq_length, emb_size) -->  (bsz, seq_length*emb_size)
        x = x.view(-1, self.seq_length*self.emb_size)
        return x
    
    
class MLP_Mixer(nn.Module):
    
    def __init__(self, seq_length,
                       num_words,
                       hidden_size1, 
                       emb_size, 
                       hidden_size2, 
                       num_classes):
        super().__init__()
        self.feature_extractor = FeatureExtractor(seq_length,num_words,hidden_size1, emb_size)
        self.classifier = MLP(seq_length*emb_size, hidden_size2, num_classes)
        
    def forward(self, x):
        x = self.feature_extractor(x)
        scores = self.classifier(x)
        return scores  
