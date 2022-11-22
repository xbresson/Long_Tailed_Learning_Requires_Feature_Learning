import torch
import torch.nn as nn
import torch.nn.functional as F
import datamodel
import models
import argparse

device = torch.device("cuda")

parser = argparse.ArgumentParser(description='neural net')

# data model
parser.add_argument('-L' ,'--L'  , type=int, default=9)
parser.add_argument('-nw','--nw' , type=int, default=150)
parser.add_argument('-nc','--nc' , type=int, default=5)
parser.add_argument('-R' ,'--R'  , type=int, default=1000)
parser.add_argument('-nspl_fam'  ,'--nspl_fam',   type=int, default=5)
parser.add_argument('-nspl_unfam','--nspl_unfam', type=int, default=3)

# neural net architecture
parser.add_argument('-h1','--hidden_size1', type=int, default=500)
parser.add_argument('-emb','--emb_size', type=int, default=10)
parser.add_argument('-h2','--hidden_size2', type=int, default=2000)

# training
parser.add_argument('-epoch','--num_epochs', type=int, default=10000000)
parser.add_argument('-bsz','--train_bsz', type=int, default=100)
parser.add_argument('-test_bsz','--test_bsz', type=int, default=100)
parser.add_argument('-lr','--lr', type=float, default=0.01)

args = parser.parse_args()


###########################
# MAKE TRAIN AND TEST SETS
##########################

fam_generator = datamodel.SequenceModel(seq_length = args.L, 
                                         num_words = args.nw,
                                         num_concepts = args.nc, 
                                         num_seq_per_class = 1, 
                                         num_classes = args.R)

unf_generator = datamodel.SequenceModel(seq_length = args.L, 
                                         num_words = args.nw,
                                         num_concepts = args.nc, 
                                         num_seq_per_class = 1, 
                                         num_classes = args.R)


train_data_fam, train_label_fam = fam_generator.generate_k_sentences_per_seq_of_concept(args.nspl_fam)
train_data_unf, train_label_unf = unf_generator.generate_k_sentences_per_seq_of_concept(args.nspl_unfam)     
test_data, test_label           = unf_generator.generate_k_sentences_per_seq_of_concept(10)
                                                    
train_data = torch.cat( [train_data_fam, train_data_unf] , dim=0)
train_label = torch.cat( [train_label_fam, train_label_unf] , dim=0)

train_size = train_data.shape[0]
test_size = test_data.shape[0]
print('size of the train set: \t', train_size, ' sentences' )
print('size of the test set: \t', test_size, ' unfamiliar sentences \n\n' )


# shuffle test data
junk = torch.randperm(test_size)
test_data = test_data[junk]
test_label = test_label[junk]

##################################    
# CREATE THE NETWORK AND OPTIMIZER
#################################
    
net = models.MLP_Mixer(seq_length   = args.L,
                       num_words    = args.nw,
                       hidden_size1 = args.hidden_size1, 
                       emb_size     = args.emb_size, 
                       hidden_size2 = args.hidden_size2, 
                       num_classes  = args.R)

net.to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(net.parameters() , lr=args.lr )

# for one-hot embedding of the words
converter = nn.Embedding(args.nw,args.nw)
converter.weight.data = torch.eye(args.nw,args.nw)
converter.to(device)


############################################################
# FUNCTIONS FOR EVALUATION ON TEST SET WITH NEAREST NEIGHBOR
############################################################

def extract_features(X):
    """
    INPUT: X: LongTensor of shape (num_data,L) 
              num_data must be a multiple of 100
              
    OUTPUT: Y: FloatTensor of shape (num_data, L*emb_size)
               Concatenation of the L words representations for eaxh sentences
    """
    
    num_data = X.shape[0]
    if num_data % 100 !=0:
        print('DATASET MUST BE A MULTIPLE OF 100')
        
    F = []
    with torch.no_grad():
        for i in range(0,num_data, 100):
            minibatch_data =  X[i:i+100].to(device)
            minibatch_data = converter(minibatch_data)
            feat = net.feature_extractor( minibatch_data ) 
            F.append(feat)
            
        Y = torch.cat(F,dim=0)
        
    return Y


def eval_on_test_set_with_nearest_nghb():
    """
    Extract features from training set and test set
    Then classify the test points by assigning to each test point the label
    of the most similar train point (measured with cosine similarity on extracted features)
    """
    
    with torch.no_grad():
        train_feat = extract_features(train_data)  
        test_feat  = extract_features(test_data)
        Gram_matrix = torch.matmul(test_feat,train_feat.t())
        predicted_labels = train_label[ torch.argmax(Gram_matrix,dim=1) ]
        num_correct = (predicted_labels == test_label).sum().item()
        error_rate = 1-num_correct/test_label.shape[0]
        
    return  error_rate 

  
######################################################
# FUNCTIONS FOR EVALUATION ON TEST SET WITH NEURAL NET
######################################################


def get_error( scores , labels ):
    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    return 1-num_matches.float()/bs   

def eval_on_test_set_with_neural_net():

    running_error=0
    num_batches=0
    
    with torch.no_grad():
        for i in range(0, test_size, args.test_bsz):
            minibatch_data =  test_data[i:i+args.test_bsz].to(device)
            minibatch_data = converter(minibatch_data)
            minibatch_label= test_label[i:i+args.test_bsz].to(device)
            scores=net( minibatch_data ) 
            error = get_error( scores , minibatch_label)
            running_error += error.item()
            num_batches+=1

    return running_error/num_batches


################
# TRAIN LOOP
################

train_loss = 0.0
train_error = 0.0

for epoch in range(args.num_epochs):
    
    shuffled_indices=torch.randperm(train_size)
    running_loss=0
    running_error=0
    num_batches_this_epoch=0

    for count in range(0,train_size, args.train_bsz):

        optimizer.zero_grad()
        indices=shuffled_indices[count:count+args.train_bsz]
        minibatch_data =  train_data[indices].to(device)
        with torch.no_grad():
            minibatch_data = converter(minibatch_data)
        minibatch_label= train_label[indices].to(device)
        scores = net( minibatch_data) 
        loss =  criterion( scores , minibatch_label) 
        loss.backward()
        optimizer.step()
        num_batches_this_epoch += 1
        
        with torch.no_grad():
            running_loss += loss.item()
            error = get_error( scores , minibatch_label)
            running_error += error.item() 
     
    train_loss = running_loss/num_batches_this_epoch
    train_error = running_error/num_batches_this_epoch
    
    
    # evaluate on test set every 10 epoch using both stragies
    if epoch % 10 == 0:
        test_error1 = eval_on_test_set_with_neural_net()
        test_error2 = eval_on_test_set_with_nearest_nghb()
        
        print( 'epoch: {} \ntrain loss: \t {:.3e} \ntrain error \t {:.2f} percent'.format( 
         epoch, train_loss, train_error*100) )
        print( 'Evaluate on unfamiliar test sentences using the neural net itself: \t {:.2f} percent of error'.format(test_error1*100))
        print( 'Evaluate on unfamiliar test sentences using a nearest neighbor rule on top of the features extracted by the neural net: \t {:.2f} percent of error'.format(test_error2*100))
        print('-----------')
        
        
        
        

