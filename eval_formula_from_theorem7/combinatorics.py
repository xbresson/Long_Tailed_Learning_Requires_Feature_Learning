import math
import scipy.special
import torch

def binomial(n,k):
    """
    Compute n choose k
    """
    return scipy.special.comb(n, k, exact=True)


def multinomial(lst):
    """
    INPUT: list or tuple of integers
    OUTPUT: multinomial value of this list
    """
    sum_of_k = lst[0]
    p = 1
    for k in lst[1:]:
        sum_of_k += k
        p = p * binomial(sum_of_k, k)
    return p



def build_stirling_triangle(n_max):
    """
    Precompute Stirling numbers of the second kind up to n_max
    this function returns a matrix St such that
    
    St[n,k] = stirling number of the second kind n,k 
    
    Due to zero indexing St is has shape (n_max+1,n_max+1)
    All non-relevant entries of the matrix are equal to zero.
    
    INPUT: n_max: (int) maximal value of n for which we precompute the stirling numbers
    OUTPUT: St:  LongTensor of shape (n_max+1,n_max+1)
    """
    St = torch.zeros(n_max+1, n_max+1, dtype = torch.int64)
    St[1,1]=1
    for n in range(2, n_max+1):
            for k in range(1,n+1):
                St[n,k] = St[n-1,k-1] + k*St[n-1,k]
    return St