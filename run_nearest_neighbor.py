import numpy as np
from sklearn import svm
import torch
import argparse

parser = argparse.ArgumentParser(description='one-shot generalization')
parser.add_argument('-gram_test','--gram_test', type=str, default='gram_test_exp2.pt')
parser.add_argument('-train_label','--train_label', type=str, default='train_label_exp2.pt')
parser.add_argument('-test_label','--test_label', type=str, default='test_label_exp2.pt')
args = parser.parse_args()

gram_test = torch.load('precomputed_gram_matrices/' + args.gram_test)
train_label = torch.load('precomputed_gram_matrices/' + args.train_label)
test_label = torch.load('precomputed_gram_matrices/' + args.test_label)

def kernel_nearest_neighbor(gram_test, train_label, test_label):
    num_test = gram_test.shape[0]
    idx_most_sim = torch.argmax(gram_test, dim=1)
    label_most_sim = train_label[idx_most_sim]
    indicator_correct = (label_most_sim == test_label).long()
    num_success = indicator_correct.sum().item() 
    return num_success/num_test

def add_noise(A):
    epsilon = 1e-4
    return A + epsilon*torch.rand(A.shape)

# Run the kernel nearest neighbor method on the parturbed gram matrix 100 times
# (use a different perturbation each time)

print('Doing 100 runs, each one with a differently perturbed Gram matrix...')

num_run = 100
acc = 0
for i in range(0,num_run):
    acc += kernel_nearest_neighbor(  add_noise(gram_test), train_label, test_label) 
acc = acc/num_run

print('accuracy on unfamiliar test sentences: {:.2f} percent'.format(acc*100))
