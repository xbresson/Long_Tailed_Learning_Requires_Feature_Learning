import torch
import math
import enumerators
from combinatorics import binomial, multinomial, build_stirling_triangle

def f(k, L, nc, sc):
    """
    Function f as defined in the paper:
    
    f(k) = nc^L * I / Multinomial(sc,sc,...,sc)
    
    where I is a sum over admissible matrices in A_k
    
    INPUT: k: list of length L+1 containing integers
           L,nc,sc: int
           
    OUPUT: float
    """
    admissible_matrices = enumerators.enumerate_admissible_matrices(nc,sc,k)
    I = 0
    for A in admissible_matrices:
        p=1
        for i in range(A.shape[0]):
            p = p * multinomial(A[i].tolist())
        I = I + p
    return  (nc**L * I) / multinomial([sc]*nc)

    

def g(k, L, nw, Stirling_triangle):
    
    """
    Function g as defined in the paper.
    
    INPUT: k: list of length L+1 containing integers
           L,nw: int
           Stirling_triangle: LongTensor containing the precomputed values of 
                              Stirling numbers of the second kind
                              Stirling_triangle[n,k] must be the stirling number n,k                    
    OUPUT: float
    """
    
    # First term of the function g.
    # We implement the version appearing in the proof of Lemma S
    # because it is numerically more stable.
    # This term count the number of forest in G_k
    
    first_term = multinomial([(j+1)*ki for j,ki in enumerate(k)])
    for j,ki in enumerate(k):
        i = j+1 # due to zero indexing
        if ki != 0 and i >=2:
            first_term *=   i**(ki*(i-2))  *  (multinomial([i]*ki)//math.factorial(ki))
    
    # Second term of the function g
    # This term count the number of (x,y) mapped to a graph with m edges
    
    m = sum( [ j*ki for j,ki in enumerate(k)] )
    second_term = 0
    for i in range(m,L+1):
        second_term += binomial(L,i) * Stirling_triangle[i,m].item() * 2**i * nw**(L-i)
    
    return first_term * second_term * math.factorial(m) / nw**(2*L)


    
def get_upperbound(L,sc,nc,t,ell):
    """
    Evaluate the formula:
    1 - sum_{k in S_ell} f(k) g(k) + max_{k in S_ell} f(k) / (t+1)
    """
    nw = sc*nc
    St = build_stirling_triangle(L)
    list_of_k = enumerators.enumerate_S_ell(L,nw,ell)
    list_of_fval = [f(k,L,nc,sc) for k in list_of_k]
    list_of_gval = [g(k,L,nw,St) for k in list_of_k]
    list_of_prod = [fval*gval for fval,gval in zip(list_of_fval,list_of_gval)]
    return 1 - sum(list_of_prod) + max(list_of_fval)/(t+1)
    
    