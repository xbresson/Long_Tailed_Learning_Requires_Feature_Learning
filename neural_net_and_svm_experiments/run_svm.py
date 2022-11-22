import torch
import torch.nn as nn
import numpy as np
from sklearn import svm
import datamodel
import argparse

parser = argparse.ArgumentParser(description='svm')

# data model
parser.add_argument('-L' ,'--L'  , type=int, default=9)
parser.add_argument('-nw','--nw' , type=int, default=150)
parser.add_argument('-nc','--nc' , type=int, default=5)
parser.add_argument('-R' ,'--R'  , type=int, default=1000)
parser.add_argument('-nspl_fam'  ,'--nspl_fam',   type=int, default=5)
parser.add_argument('-nspl_unfam','--nspl_unfam', type=int, default=1)

parser.add_argument('-C' ,'--C'  , type=float, default=1.0)
parser.add_argument('-gamma' ,'--gamma'  , type=float, default=1.0)
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


#######################
# PREPROCESS THE DATA
########################

# One-hot-Embedding
converter = nn.Embedding(args.nw,args.nw)
converter.weight.data = torch.eye(args.nw,args.nw)

with torch.no_grad():
    train_data = converter(train_data)
    test_data = converter(test_data)
    
train_data = train_data.view(train_size , args.L*args.nw)
test_data = test_data.view(test_size , args.L*args.nw)

# Normalize
p = 1/args.nw
q=1-p
train_data = (train_data - p)/ (p*q)**0.5
test_data = (test_data - p)/ (p*q)**0.5

#######################
# SCIKIT-LEARN 
########################

print('Running scikit-learn SVM on features extracted by psi_{one-hot} (it takes ~10 minutes)')
clf = svm.SVC(C=args.C , kernel='linear')
clf.fit(train_data.numpy(), train_label.numpy())
test_accuracy = clf.score(test_data.numpy(), test_label.numpy())
print('accuracy on unfamiliar test sentences: \t {:.2f} percent \n\n'.format(test_accuracy*100 ) )

print('Running scikit-learn SVM with GAUSSIAN KERNEL on features extracted by psi_{one-hot} (it takes ~10 minutes)')
clf = svm.SVC(C=args.C ,gamma=args.gamma, kernel='rbf')
clf.fit(train_data.numpy(), train_label.numpy())
test_accuracy = clf.score(test_data.numpy(), test_label.numpy())
print('accuracy on unfamiliar test sentences: \t {:.2f} percent'.format(test_accuracy*100 ) )


        