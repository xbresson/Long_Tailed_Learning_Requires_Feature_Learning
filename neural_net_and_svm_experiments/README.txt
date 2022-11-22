#####################################
# EXPERIMENTS WITH THE NEURAL NETWORK
#####################################

The command below will generate a train and a test set from our data model with parameters set to L=9, nw=150, nc=5, R=1000, nspl=8 and n^*=3 (8 sentences per category in the training set: 5 are familiar and 3 are unfamiliar). Then it will train the neural network in Figure 2. At test time both evaluation strategies will be used (i.e. the neural net itself and the nearest neighbor classification rule on the top of the learned features). The test set contains 10R unfamiliar sentences (10 sentences per category).

python run_neural_net.py -L 9 -nw 150 -nc 5 -R 1000 -nspl_fam 5 -nspl_unfam 3

To try different size for the layer of the neural network, and different learning rate, do

python run_neural_net.py -L 9 -nw 150 -nc 5 -R 1000 -nspl_fam 5 -nspl_unfam 3 -h1 100 -emb 5 -h2 1000 -lr 0.1

where h1 and emb are the hidden and ouput sizes of the first MLP, and h2 is the hidden size of the second MLP.


#######################
# EXPERIMENTS WITH SVM 
#######################

The following command will generate a train set and a test set. It will then use SCIKIT-LEARN to fit a plain SVM and a SVM with gaussian kernel on the features extracted by psi_{one-hot}. The test set contains 10R unfamiliar sentences (10 sentences per category). Gamma is the parameter involved in the definition of the Gaussian kernel. 

python run_svm.py -L 9 -nw 150 -nc 5 -R 1000 -nspl_fam 5 -nspl_unfam 3 -gamma 0.1
