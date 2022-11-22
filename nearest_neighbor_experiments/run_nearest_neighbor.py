import torch
import datamodel
import kernel_routines
import math
import argparse


##############
# ARGPARSE
##############

parser = argparse.ArgumentParser(description='nearest_neighbor')

parser.add_argument('-L' ,'--L'  , type=int, default=9)
parser.add_argument('-nw','--nw' , type=int, default=150)
parser.add_argument('-nc','--nc' , type=int, default=5)
parser.add_argument('-nspl_fam'  ,'--nspl_fam',   type=int, default=5)
parser.add_argument('-nspl_unfam','--nspl_unfam', type=int, default=3)
parser.add_argument('-R' ,'--R'  , type=int, default=1000)
parser.add_argument('-test_size' ,'--test_size'  , type=int, default=100)

args = parser.parse_args()


#############################
# GENERATE TRAIN AND TEST SET
#############################

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
test_data_unf, test_label_unf   = unf_generator.generate_k_sentences_per_seq_of_concept(10)

train_data = torch.cat( [train_data_fam, train_data_unf] , dim=0)
train_label = torch.cat( [train_label_fam, train_label_unf] , dim=0)

# truncate test set
junk = torch.randperm(test_data_unf.shape[0])
test_data_unf = test_data_unf[junk[:args.test_size]]
test_label_unf = test_label_unf[junk[:args.test_size]]

ntrain = train_data.shape[0]
ntest = test_data_unf.shape[0]
print('\nSize of the training set: ', ntrain, ' sentences')
print('Size of the test set: ', ntest, ' unfamiliar sentences')


###########################################
# COMPUTE GRAM TEST MATRIX FOR PSI_STAR
###########################################
print('\nCompute the dot products < psi_star(x) , psi_star(y)> for all x in the test set and all y in the train set')
gram_test_star    = torch.zeros(ntest,ntrain).double()
for i in range(ntest):
    for j in range(ntrain):
        gram_test_star[i,j] = kernel_routines.moon_kernel(
                               test_data_unf[i],train_data[j], args.nw, args.nc)
    print('num of test sentences processed: {} out of {}'.format(i+1, ntest))


############################################
# COMPUTE GRAM TEST MATRIX FOR PSI_{ONE-HOT}
############################################
print('\nCompute the dot products < psi_{one-hot}(x) , psi_{one-hot}(y)> for all x in the test set and all y in the train set')
gram_test_one_hot = torch.zeros(ntest, ntrain )
for i in range(ntest):
    for j in range(ntrain):
        gram_test_one_hot[i,j] = (test_data_unf[i] == train_data[j]).sum().item()
    print('num of test sentences processed: {} out of {}'.format(i+1, ntest))



##############################
# NEAREST NEIGHBOR ALGORITHM
##############################

def kernel_nearest_neighbor(gram_test, train_label, test_label):
    idx_most_sim = torch.argmax(gram_test, dim=1)
    label_most_sim = train_label[idx_most_sim]
    indicator_correct = (label_most_sim == test_label).long()
    num_success = indicator_correct.sum().item() 
    return num_success

def add_noise(A):
    epsilon = 1e-4
    return A + epsilon*torch.rand(A.shape)

#######################
# RUN NEAREST NEIGHBOR 
#######################
# note that we add noise to break ties
num_correct_star = kernel_nearest_neighbor(  add_noise(gram_test_star), train_label, test_label_unf) 
num_correct_one_hot = kernel_nearest_neighbor(  add_noise(gram_test_one_hot), train_label, test_label_unf) 

print('\nNEAREST NEIGHBOR ON FEATURES EXTRACTED BY PSI_STAR')
print('Number of unfamiliar test sentences correctly classified:\t {} out of {} '.format(num_correct_star,ntest) )

print('\nNEAREST NEIGHBOR ON FEATURES EXTRACTED BY PSI_{ONE-HOT}')
num_correct = kernel_nearest_neighbor(  add_noise(gram_test_one_hot), train_label, test_label_unf) 
print('Number of unfamiliar test sentences correctly classified:\t {} out of {} '.format(num_correct_one_hot,ntest) )


