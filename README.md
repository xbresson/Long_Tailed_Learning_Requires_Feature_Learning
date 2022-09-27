# Long-tailed Learning Requires Feature Learning


<br>

## Python environment

conda env create -f environment.yml

conda activate pytorch_env

<br>


## Uncompressed precomputed gram matrices

unzip precomputed_gram_matrices.zip

<br>

## Neural Net 

To reproduce the results from the first experiment type:

python run_neural_net.py -L 9 -nw 150 -nc 5 -R 1000 -nspl_fam 5 -nspl_unfam 1

This will generate a training set containing R=1000 categories. Each category contains 5 familiar sentences and 1 unfamiliar sentence. So overall the training set contains 5000 familiar sentences and 1000 unfamiliar sentences. The code will also generate a test set containing 1000 unfamiliar sentences. The neural net described in the paper is then trained on the train set. The error on the test set is printed every 50 epoch.

To reproduce the result from the second experiment type:

python run_neural_net.py -L 15 -nw 30 -nc 5 -R 1000 -nspl_fam 5 -nspl_unfam 5

This will generate a train set containing 5000 familiar sentences and 5000 unfamiliar sentences. It will also generate a test set containing 5000 unfamiliar sentences.

To try different sizes for the layers of the neural network, and different learning rate, type:

python run_neural_net.py -L 15 -nw 30 -nc 5 -R 1000 -nspl_fam 5 -nspl_unfam 5 -h1 100 -emb 5 -h2 1000 -lr 0.1

where h1 and emb are the hidden and ouput sizes of the first MLP, and h2 is the hidden size of the second MLP.

<br>

## SVM with RBF Kernel

To reproduce the results from the first experiment type:

python run_rbf.py -L 9 -nw 150 -nc 5 -R 1000 -nspl_fam 5 -nspl_unfam 1 -gamma 0.01 -C 1.0 

It will generate a train and test set as described above for the neural net, then it will call the SVC function from Scikit-learn in order to run a soft, multiclass SVM with RBF kernel.

To reproduce the results from the second experiment type:

python run_rbf.py -L 15 -nw 30 -nc 5 -R 1000 -nspl_fam 5 -nspl_unfam 5 -gamma 0.75 -C 1.0 

<br>

## SVM with Optimal Kernel K-STAR

To reproduce the result from the second experiment type:

python run_svm.py -gram_train gram_train_exp2.pt -gram_test gram_test_exp2.pt -train_label train_label_exp2.pt -test_label test_label_exp2.pt -log_alpha 7.2 -C 1.0

The file gram_train_exp2.pt in the precomputed_gram_matrices folder contains a 10000-by-10000 matrix A in which

A_ij = | { phi: phi(x^train_i) = phi(x^train_j) } | / factorial(nc)   (1)

where x^train_i is the ith point of the training set. The training set contains 10,000 data points that were generated according to the data model used in the second experiment. The file gram_test_exp2.pt contains a 5000-by-5000 matrix B in which

B_ij = | { phi: phi(x^test_i) = phi(x^train_j) } | / factorial(nc)    (2)

where x^test_i is the ith point of the test set. The test set contains 5,000 unfamiliar sentences that were generated according to the data model used in the second experiment. The files train_label_exp2.pt and test_label_exp2.pt contains the labels of the train and test points. Using the matrices A and B, the code will create the gram matrices described in the paper (with logarithmic transform), then it will feed these Gram matrices to the SVC function from Scikit-learn in order to run a soft, multiclass SVM. Remark: the code also compute the min eigenvalue of the train Gram matrix to check that it is positive.

To reproduce the result from the first experiment type:

python run_svm.py -gram_train gram_train_exp1.pt -gram_test gram_test_exp1.pt -train_label train_label_exp1.pt -test_label test_label_exp1.pt -log_alpha 1.0 -C 1.0

The file gram_train_exp1.pt in the precomputed_gram_matrices folder contains a 6000-by-6000 matrix A in which

A_ij = Kstar(x^train_i, x^train_j) / Kstar(xhat, yhat)     (3)

where (xhat,yhat) is the pair of sentences minimizing the optimal kernel (see the paper.) We did not use formula (1) because it leads to numbers which are too large (indeed in experiment 1 the vocabulary is much larger and as a consequence the number |{phi:phi(x^train_i)=phi(x^train_j)}| is huge ). The factor 1/Kstar(xhat, yhat) is here to avoid overflow when computing the gram matrices. Similarly the file gram_test_exp1.pt contains a 1000-by-1000 matrix B in which

B_ij = Kstar(x^test_i, x^train_j) / Kstar(xhat, yhat)     (4)

<br>

## Neareast-Neighbor with Optimal Kernel K-STAR


To reproduce the result from the second experiment type:

python run_nearest_neighbor.py -gram_test gram_test_exp2.pt -train_label train_label_exp2.pt -test_label test_label_exp2.pt 

It will run the kernel-nearest-neighbor method with the test gram matrix described in formula (2).

To reproduce the result from the first experiment type:

python run_nearest_neighbor.py -gram_test gram_test_exp1.pt -train_label train_label_exp1.pt -test_label test_label_exp1.pt

It will run the kernel-nearest-neighbor method with the test gram matrix described in formula (4).

<br>



