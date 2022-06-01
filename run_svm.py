import numpy as np
from sklearn import svm
import torch
import argparse

parser = argparse.ArgumentParser(description='one-shot generalization')
parser.add_argument('-log_alpha','--log_alpha', type=float, default=7.2)
parser.add_argument('-C','--C', type=float, default=1.0)
parser.add_argument('-tol','--tol', type=float, default=0.0001)
parser.add_argument('-gram_train','--gram_train', type=str, default='gram_train_exp2.pt')
parser.add_argument('-gram_test','--gram_test', type=str, default='gram_test_exp2.pt')
parser.add_argument('-train_label','--train_label', type=str, default='train_label_exp2.pt')
parser.add_argument('-test_label','--test_label', type=str, default='test_label_exp2.pt')
args = parser.parse_args()

gram_train = torch.load('precomputed_gram_matrices/' + args.gram_train)
gram_test = torch.load('precomputed_gram_matrices/' + args.gram_test)
train_label = torch.load('precomputed_gram_matrices/' + args.train_label)
test_label = torch.load('precomputed_gram_matrices/' + args.test_label)
ntrain = train_label.shape[0]
ntest = test_label.shape[0]

# The training gram matrix was stored as upper triangular
# Fill the lower triangular part so that it is symmetric
gram_train = gram_train + gram_train.t()
for i in range(ntrain):
    gram_train[i,i] = gram_train[i,i]/2 
    
# Logarithmic transform of the kernel 
def kernel_transform(K):
    alpha = 10**args.log_alpha
    return torch.log10( 1 + K/alpha)

gram_train = kernel_transform(gram_train)
gram_test  = kernel_transform(gram_test)

# checking the eigenvalue of the Gram train matrix
print('Computing the eigenvalues of the Gram Train Matrix (it takes a minute.)')
print('(lambda min shoudld be positive -- if not increase the value of log_alpha)')
eigval, eigvec = np.linalg.eigh( gram_train.numpy() )
lambda_min = eigval.min()
lambda_max = eigval.max()
print('lambda_min = {:.2f}, \t lambda_max = {:.2f}'.format(lambda_min, lambda_max))

# scikit-learn
clf = svm.SVC(kernel='precomputed',C=args.C,verbose=False, tol=args.tol, break_ties=False, shrinking = True)
print('\nRunning scikit-learn SVC function (it takes a few minutes)')
clf.fit(gram_train.numpy(), train_label.numpy())
test_accuracy = clf.score(gram_test.numpy(), test_label.numpy())
print('test accuracy on unfamiliar test sentences: {:.2f} percent'.format(test_accuracy*100))