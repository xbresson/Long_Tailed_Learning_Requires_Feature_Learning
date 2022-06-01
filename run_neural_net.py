import torch
import torch.nn as nn
import torch.nn.functional as F
import datamodel
import model
import argparse

device = torch.device("cuda")

parser = argparse.ArgumentParser(description='one-shot generalization')

# data model
parser.add_argument('-L' ,'--L'  , type=int, default=9)
parser.add_argument('-nw','--nw' , type=int, default=150)
parser.add_argument('-nc','--nc' , type=int, default=5)
parser.add_argument('-R' ,'--R'  , type=int, default=1000)
parser.add_argument('-nspl_fam'  ,'--nspl_fam',   type=int, default=5)
parser.add_argument('-nspl_unfam','--nspl_unfam', type=int, default=1)

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
                                         num_seq_per_class = args.nspl_unfam, 
                                         num_classes = args.R)

train_data_fam, train_label_fam = fam_generator.generate_k_sentences_per_seq_of_concept(args.nspl_fam)
train_data_unf, train_label_unf = unf_generator.generate_k_sentences_per_seq_of_concept(1)               
test_data, test_label           = unf_generator.generate_k_sentences_per_seq_of_concept(1)
                                                    
train_data = torch.cat( [train_data_fam, train_data_unf] , dim=0)
train_label = torch.cat( [train_label_fam, train_label_unf] , dim=0)

train_size = train_data.shape[0]
test_size = test_data.shape[0]

print('number of classes = ', args.R)
print('total number of familiar train sentences = ', train_data_fam.shape[0])
print('total number of unfamiliar train sentences = ', train_data_unf.shape[0])
print('total number of unfamiliar test sentence (no familiar test sentence!) = ', test_data.shape[0] )


##################################    
# CREATE THE NETWORK AND OPTIMIZER
#################################
    
net = model.Simple_Net(seq_length   = args.L,
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

  
################
# EVAL FUNCTION
################

def get_error( scores , labels ):
    bs=scores.size(0)
    predicted_labels = scores.argmax(dim=1)
    indicator = (predicted_labels == labels)
    num_matches=indicator.sum()
    return 1-num_matches.float()/bs   

def eval_on_test_set():

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
    
    if epoch % 50 == 0:
        test_error = eval_on_test_set()
        print( 'epoch {} \t train loss: {:.3e} \t train error (on fam and unfam sentences): {:.2f}% \t test error on unfamiliar sentences only: {:.2f}%'.format( 
        epoch, train_loss, train_error*100, test_error*100) )
        