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
    
    
class Simple_Net(nn.Module):
    
    def __init__(self, seq_length,
                       num_words,
                       hidden_size1, 
                       emb_size, 
                       hidden_size2, 
                       num_classes):

        super().__init__()
    
        self.layer1 = MLP(num_words,hidden_size1, emb_size)
        self.layer2 = MLP(seq_length*emb_size, hidden_size2, num_classes)
        self.normalization = nn.LayerNorm(emb_size)
        self.seq_length = seq_length
        self.emb_size = emb_size
    
    def forward(self, x):
        
        # (bsz, seq_length, num_words) --> (bsz, seq_length, emb_size)
        x = self.layer1(x) 
        
        # Layer norm
        x = self.normalization(x)
       
        #(bsz, seq_length, emb_size) -->  (bsz, seq_length*emb_size)
        x = x.view(-1, self.seq_length*self.emb_size)

        # (bsz, seq_length*emb_Size) --> (bsz, num_classes)
        x = self.layer2(x)
        return x  
