# Long-tailed Learning Requires Feature Learning

<br>


# Citation

```
@misc{laurent2023featlearning,
      title={Long-Tailed Learning Requires Feature Learning}, 
      author={Thomas Laurent and James von Brecht and Xavier Bresson},
      year={2023},
      booktitle={International Conference on Learning Representations (ICLR)}
}
```


<br>

## Python environment

conda env create -f environment.yml

conda activate pytorch_env

<br>

## Experiments

The folder eval_formula_from_theorem7 contains a code that evaluates the combinatorial formula between square bracket appearing in Theorem 7. This code will display inequality (105) for every $2 \leq \ell \leq L$ (with numerical values for the constant involved). In particular it will display inequality (10) when the parameters are set to $L=9,$ $n_w=150,$ $n_c=5$ and $R=1000$.

<br>

The folder neural_net_and_svm_experiments contains codes that will run the neural network in Figure 2 on our data model, as well as a plain SVM or a SVM with Gaussian kernel on the top of features extracted by $\psi_\textrm{one-hot}$ (1st, 2nd, 7th and 8th row of the table).

<br>

The folder nearest_neighbor_experiments contains codes to run a nearest neighbor classification rule on the top of features extracted by $\psi^\star$ and $\psi_\textrm{one-hot}$ (3rd and 4th row of the table).

<br>

