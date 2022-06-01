import torch
import torch.nn as nn
import numpy as np
from sklearn import svm
import datamodel
import argparse


parser = argparse.ArgumentParser(description='kernelized SVM with RBF kernel')

# data model
parser.add_argument('-L' ,'--L'  , type=int, default=15)
parser.add_argument('-nw','--nw' , type=int, default=30)
parser.add_argument('-nc','--nc' , type=int, default=5)
parser.add_argument('-R' ,'--R'  , type=int, default=1000)
parser.add_argument('-nspl_fam'  ,'--nspl_fam',   type=int, default=5)
parser.add_argument('-nspl_unfam','--nspl_unfam', type=int, default=5)

# hyperparameters for SVM with Gaussian kernel
parser.add_argument('-gamma','--gamma', type=float, default=0.75)
parser.add_argument('-C','--C', type=float, default=1.0)
parser.add_argument('-tol','--tol', type=float, default=0.0001)

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

clf = svm.SVC(C=args.C,kernel='rbf', verbose=False, tol=args.tol, break_ties=False, gamma=args.gamma)
print('\nRunning scikit-learn SVC function (it takes 2-5 minutes)')
clf.fit(train_data.numpy(), train_label.numpy())
test_accuracy = clf.score(test_data.numpy(), test_label.numpy())
print('test accuracy on unfamiliar test sentences: {:.2f} percent'.format(test_accuracy*100))