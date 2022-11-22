import itertools
import torch
import numpy as np
from multinomial_routines import multinomial, multinomial_star
from partition_routines import get_all_integer_partitions
from graph_routines import get_mu_nu



class Unique_Perms:
    """
    Take as input a vector v of length nc containing integers
    and compute its multinomial value as well as all unique permutations
    
    To be more precise, Unique_Perms(v) is an object with 
    three attributs and no method.
    To initialize it we give a LongTensor v of shape (nc,)
    Then the following tree attribute are created:
    
    self.v:                      Longtensor of shape (nc,)
                                 This is the vector used to initialize the equivalence class
                                   
    self.perms:                  list of LongTensors of shape (nc,)
                                 These are all the permutations of v.
                                 There are NO repeats in this list
                                   
    self.mult:                   Multinomial value of v (and all its permutations)
                                   
                                   
    """
    
    def __init__(self,v):
        
        assert v.dtype == torch.int64 , 'v should have dtype int64'
        
        self.v = v
        
        # convert the tensor to a tuple 
        # then compute unique permutations using itertool and set
        # then convert back to tensor
        v_tpl = tuple( v.tolist() )
        unique_perms_of_v = list(set(itertools.permutations(v_tpl)))
        self.perms = [ torch.tensor(u) for u in unique_perms_of_v]
        
        # Compute multinomial value
        self.mult = multinomial(v_tpl)


        
def phi(list_of_tensor, mu, sc):
    """
    list_of_tensors: a list containing m LongTensors of shape (nc,)
    mu: LongTensor of shape (m,).
    sc: int
    """
    Bt = torch.stack( list_of_tensor , dim = 1)
    v = sc  -  torch.mv(Bt,mu)
    return v



def Gamma(list_of_v, mu, sc ):
    """
    INPUT: 
    list_of_v:  list [v_1, ..., v_m] where v_i is an object of the type Unique_Perms
    mu: LongTensor of shape (m,). mu_i is the size of connected components of type i
    sc: int
    
    OUTPUT: int
    """
    
    list_of_mult  = [ v.mult  for v in list_of_v]
    list_of_perms = [ v.perms for v in list_of_v]
    
    # Find the vector v with largest number of unique permutations 
    num_perms = [ len(perm) for perm in list_of_perms]
    i0 = np.argmax(num_perms)
    size0 = num_perms[i0]
    
    # replace the the largest list with a list containing a single element
    list_of_perms[i0] = list_of_perms[i0][0:1] 
    
    # Use itertools
    s = 0
    for tpl_of_tensors in itertools.product(*list_of_perms):
        v = phi( tpl_of_tensors, mu, sc)
        s = s + multinomial_star(v.tolist())
        
    myprod = 1
    for ii in list_of_mult:
        myprod = myprod * ii
        
    #return size0 * s * np.prod(list_of_mult)
    return size0 * s * myprod



def Compute_I(nc, sc, mu, nu):
    """
    This function compute I(G) as defined in the paper,
    where G is a graph containing
               nu_1 components of size mu_1
                    .
                    .
                    .
               nu_m components of size mu_m
               
    The vectors mu and nu describe the nontrivial components only,
    so mu_1,..., mu_m >= 2
 
    INPUTS:
    nc and sc: int. The number and the size of concepts
    mu and nu: LongTensors of shape (m,) 
               that describes the number of non-trivial components in the graph G
               
    OUTPUT: 
    integer I(G)                 
    """
    
    assert torch.all(mu>1).item() , 'all components should have size >= 2'
    assert torch.all(nu>=1).item() , 'all nu_i must be >= 1'
    assert mu.shape[0] == nu.shape[0], 'mu and nu must have same length'
    assert mu.dtype == torch.int64
    assert nu.dtype == torch.int64
    assert type(nc) == int
    assert type(sc) == int
    
    # compute the integer partitions up to nu_max
    nu_max = nu.max().item()
    INT_PART = get_all_integer_partitions(nu_max=nu_max,n=nc)
    m = mu.shape[0]

    # Z is a list of objects
    list_of_Z = []
    for i in range(m):
        Z = [Unique_Perms(v) for v in INT_PART[nu[i].item()] ]
        list_of_Z.append( Z )
        
    s = 0
    for tpl_of_v in itertools.product(*list_of_Z):
        s+=Gamma(tpl_of_v, mu, sc )
         
    return s


def optimal_kernel(x, y, nw, nc):
    """
    Compute the number of equipartition phi 
    that render sentences x and y conceptually equivalent:
    
    |{phi in Phi: phi(x_ell)= phi(y_ell) for all 1<= ell <= L}|
    
    Note that there is no normalization factor 1/(|Phi|*sc^L)
    
    x,y: LongTensor of shape (L,) with entries in {0,1,..., nw-1}
    nw, nc: int (num_words must be divisible by num_concepts)
    """
    
    assert nw % nc == 0, 'num_words must be divisible by num_concepts'
    sc = nw // nc
  
    if torch.equal(x,y):
        number_of_partitions = multinomial( [sc]*nc ) 
    else:
        mu, nu = get_mu_nu(x,y)
        number_of_partitions = Compute_I(nc, sc, mu, nu)
       
    return number_of_partitions


def moon_kernel(x, y, nw, nc):
    """
    Compute the number of equipartition phi 
    that render sentences x and y conceptually equivalent,
    then multiply this number by nc^L / |Phi|
    
    nc^L * |{phi in Phi: phi(x_ell)= phi(y_ell) for all 1<= ell <= L}| / |Phi|
    
    Note that this is a different normalization than the one used for K^star.
    This normalization leads to values which are O(1)
    
    x,y: LongTensor of shape (L,) with entries in {0,1,..., nw-1}
    nw, nc: int (num_words must be divisible by num_concepts)
    """
    
    assert nw % nc == 0, 'num_words must be divisible by num_concepts'
    sc = nw // nc
    L = x.shape[0]
    
    PHI = multinomial([sc]*nc)
    cst = nc**L
    
    if torch.equal(x,y):
        frac = 1 
    else:
        mu, nu = get_mu_nu(x,y)
        number_of_partitions = Compute_I(nc, sc, mu, nu)
        frac = number_of_partitions/PHI
       
    return cst * frac
