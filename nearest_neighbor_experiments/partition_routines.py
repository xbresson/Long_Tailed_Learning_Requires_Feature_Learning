import torch


def keep_sorted_rows_only(A):
    """
    Delete the rows which are not sorted (in ascending order)
    
    INPUT: 
    A: LongTensor of shape (m,n)
    
    OUTPUT: 
    B: LongTensor of shape (m-k,n) where k is the number of deleted rows 
    """
    condition = torch.diff(A,dim=1) >= 0
    row_cond = condition.all(dim=1)
    B = A[row_cond, :]
    return B


def compute_next_partition(Q,n):
    """
    Given the set:
    
    Q_{nu,n} = { u in N_0^n : sum_i u = nu  and u_1 <= u_2 <= ... <= u_n }
    
    This function compute the set
    
    Q_{nu+1,n} = { u in N_0^n : sum_i u = nu+1  and u_1 <= u_2 <= ... <= u_n }
    
    INPUT
    Q is a list of  LongTensors of shape (n,)
    All these tensor must be sorted and sum to nu
    
    OUTPUT
    Q_next: list of sorted LongTensors that sum to nu+1
    """
    
    I = torch.eye(n).long()
    
    # A is a matrix whose rows contains 
    # all the vectors of the set  Q + S
    # where S = {e_1, e_2, ..., e_n}
    A = torch.cat( [I+u for u in Q] , dim = 0  )
    
    # we then get rid of all the rows which are not sorted
    A = keep_sorted_rows_only(A)
    
    # we then get rid of all the duplicated rows
    A = torch.unique(A,dim=0)
    
    # we then unpack the rows of A into a list of LongTensors
    Q_next = [A[i] for i in range(A.shape[0])]
    return Q_next


def get_all_integer_partitions(nu_max,n):
    """
    This function compute all the sets
    
    Q_{nu,n} = { u in N_0^n : sum_i u = nu  and u_1 <= u_2 <= ... <= u_n }
    
    for nu = 0, 1, 2, ..., nu_max
    
    OUTPUT: partition is a list of length nu_max+1
            partition[nu] provides the set Q_{nu,n} 
            It is a list of LongTensors of shape (n,)
    
    """
   
    Q = [torch.zeros(n,dtype = torch.int64)] 
    partition = [Q]
                   
    for i in range(1,nu_max+1):
        Q = partition[-1]
        Qnext = compute_next_partition(Q,n)
        partition.append(Qnext)
        
    return partition