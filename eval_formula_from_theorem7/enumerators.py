import torch
import simplex 
import itertools


def get_COO_format(v):
    """
    Compute the coordinate format of the vector v
    
    For example, if v = [5,0,0,7,9] then
    idx = [0,3,4]
    val = [5,7,9]
    
    INPUT: v (LongTensor)
    OUTPUT: idx,val LongTensors of same length
    """
    idx = torch.nonzero(v).view(-1)
    val = v[idx]
    return idx,val


def enumerate_admissible_matrices(nc,sc,k):
    """
    This function compute the set A_k of admissible assignement matrices
    as defined in the paper. Note that these are (L+1)-by-nc matrices.
    However we ignore the rows filled with zeros since they do not contribute to the formula.
    Therefore this function returns a list of R-by-nc matrices where R is the number of nonzero
    entries in the vector k. 
    
    We first compute rows A_2,...,A_R (corresponding to components of size >=1)
    We then use the column sum condition to get the first row (corresponding to components of size 1)
    
    INPUT: nc, sc: integers
           k: list of length L+1 containing the number of components of each size
    OUTPUT: admissible matrices: list of LongTensors that have shape (R,nc) 
                                 where R is the number of nonzero entries in k
    """
    # Get COO format of the vector [k2,k3,...,k_{L+1}]
    # comp_sz and comp_num are list of length R-1
    # that describe the number and size the non trivial components present in the graph.
    idx, val = get_COO_format(torch.tensor(k[1:]))
    comp_sz = idx+2  
    comp_num = val   

    # rows is a list of length R-1
    # rows[i] describe all the possible ways to assign the components of size comp_sz[i] to the nc bins
    rows = [ simplex.simplex(nc,s) for s in comp_num.tolist()]
    
    # We now construct the admissible matrices
    # The row corresponding to components of size 1 is infered from the other rows
    admissible_matrices = []
    for matrix in itertools.product(*rows):
        A = torch.tensor(matrix)
        v = sc  -  torch.mv(A.t(),comp_sz) # get row for components of size 1
        if torch.all(v>=0).item():
            admissible_matrices.append(torch.cat([v.view(1,nc),A]))
    return admissible_matrices


def enumerate_Sell_backslash_Sell_plus_one(L,nw,ell):
    """
    Recall that the sets S_ell defined in the paper are nested as follow:
    
    S_L subset S_{L-1} subset S_{L-2} subset ... subset S_0
    
    This function compute the set
    
                          S_ell \ S_{ell+1}

    Looking at the definition of the sets S_ell, this is equivalent to finding
    all the vectors k = [k1,k2,...,k_{L+1}] that satisfy:
    
                  k_i >= 0
                  1*k1 + 2*k2 + 3*k3 + ... + (L+1)*k_{L+1} = nw
                  0*k1 + 1*k2 + 2*k3 + ... +   L  *k_{L+1} = ell
                  
   or, equivalently:
    
                  k_i >= 0
                  k1 + k2 + k3 + ... + k_{L+1} = nw-ell
                  k2 + 2*k3 + 3*k4 + ... + L*k_{L+1} = ell
                  
    To do this we start by solving k2 + 2*k3 + 3*k4 + ... + L*k_{L+1} = b
    We then compute k1 = nw - b - ( k2 + k3 + ... + k_{L+1} )
    
    INPUT: L: length of the sequence
           nw: number of word in the vocabulary (or number of vertices in the graph)
           ell: int
           
    OUTPUT: list of list. Each list has the form [k1,k2,...,k_{L+1}]
    """
    a = list(range(2,L+1)) # a = [2,3,...,L]
    sln = simplex.weighted_simplex(a,ell) # solve k2 + 2*k3 + 3*k4 + ... + L*k_{L+1} = ell
    S = []
    for k2_to_klast in sln: 
        k1 = nw - ell - sum(k2_to_klast)  # compute k1
        if k1 >= 0:
            S.append( [k1]+k2_to_klast) 
    return S


def enumerate_S_ell(L,nw,ell):
    """
    Find all the vectors k = [k1,k2,...,k_{L+1}] that satisfy:
    
                         k_i >= 0
                         1*k1 + 2*k2 + 3*k3 + ... + (L+1)*k_{L+1} = nw
                  ell <= 0*k1 + 1*k2 + 2*k3 + ... +   L  *k_{L+1} <= L
    
    INPUT: L: length of the sequence
           nw: number of word in the vovabulary (or number of vertices in the graph)
           b: int
           
    OUTPUT: list of list. Each list has the form [k1,k2,...,k_{L+1}]
    """
    S_ell = []
    for b in range(ell,L+1):
        S_ell.extend(enumerate_Sell_backslash_Sell_plus_one(L,nw,b))
    return S_ell

