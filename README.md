# One-Shot Generalization

The following command will run the neural network on our data model with
parameters set to L=9, nw=150, nc=5, R=1000 and nspl=6 (5 familiar sentences and 1 unfamiliar sentence per category). These are the parameters corresponding to the main Theorem of the paper. The neural net has the architecture described in the paper.

conda env create -f environment.yml
conda activate pytorch_env

python main.py -L 9 -nw 150 -nc 5 -R 1000 -nspl_fam 5 -nspl_unfam 1

The following command will run the neural network on the data model described in the second experiment (each category contains 5 familiar sentences and 5 unfamiliar sentences)

python main.py -L 15 -nw 30 -nc 5 -R 1000 -nspl_fam 5 -nspl_unfam 5

To try different size for the layer of the neural network, and different learning rate, do

python main.py -L 15 -nw 30 -nc 5 -R 1000 -nspl_fam 5 -nspl_unfam 5 -h1 100 -emb 5 -h2 1000 -lr 0.1

where h1 and emb are the hidden and ouput sizes of the first MLP, and h2 is the hidden size of the second MLP.