The command 

python run_nearest_neighbor.py -L 9 -nw 50 -nc 5 -R 1000 -nspl_fam 5 -nspl_unfam 3 -test_size 100

will do the following:

1) It generates a train set from our data model with parameters set to $L=9$, $n_w=50$, $n_c=5$, $R=1000$, $n_\textrm{spl}=8$ and $n^\star=3$ (8 sentences per category in the training set: 5 are familiar and 3 are unfamiliar). So in total the train set contains 8000 sentences.

2) it generates a test set containing 100 unfamiliar sentences.

3) it computes the dot products $\langle \psi^\star(x) , \psi^\star(y) \rangle$ for all $x$ in the test set and all $y$ in the train set. This is an expensive combinatorial computation. We recommend choosing a small test set (e.g. 100 sentences in the set) so that the computation can be done in less than an 1 hour.

4) it computes the dot products $\langle \psi_\textrm{one-hot}(x) , \psi_\textrm{one-hot}(y) \rangle$ for all $x$ in the test set and all $y$ in the train set. 

5) Use the nearest neighbor classification rule on both types of features ($\psi^\star$ and $\psi_\textrm{one-hot}$) to classify the test sentences. With the chosen parameters, around 5 unfamiliar test sentences will be correctly classified out of the 100 unfamiliar test sentences.



