import torch


def integers_histogram(v):
    """
    Make an histogram indicating the count of each entry of v
    example:
              INPUT:    v = [15, 20, 15, 13, 13 ,15 , 15]
              OUTPUT:   hist = {15: 4 ,   20: 1  ,   13: 2}
              
    INPUT: v: LongTensor of shape (n)
    Output: hist: dictionary whose key and values are int
    """
    
    hist = {}
    for i in range(v.shape[0]):
        this_key = v[i].item()
        if this_key in hist:
            hist[ this_key ] +=1
        else:
            hist[ this_key ] = 1
    return hist


def create_adjacency(edge_start, edge_end):
    """
    
    edge_start and edge_end contains the index of vertices to be connected
    Let say there are n0 distinct vertices in (edge_start union edge_end).
    This function returns the n0-by-n0 adjacency matrix showing the connections between these vertices.
    The matrix A is indexed so that the order of the vertices is preserved
  
    INPUT
    edge_start and edge_end: LongTensors of shape (m,) containing integers 
                             The edges of the graph are the pairs 
                                     ( edge_start[i] , edge_end[i] )
                             for i = 0,...,m-1  
                                                 
    OUTPUT:
    A: LongTensor of shape (n0,n0)
       Adjacency matrix of the subgraph

    """   
    vertices = torch.unique(torch.cat([edge_start, edge_end]))
    max_idx = vertices.max().item()
    
    # add edge to a graph with max_idx vertices
    A = torch.eye(max_idx+1)
    A[edge_start,edge_end]=1
    
    # make the graph symmetric and binary
    A = A + A.t()
    A = (A>0).long()
    
    # Get rid of the vertices of degree zero
    A = A[vertices,:]
    A = A[:,vertices]
    
    return A


def get_connected_components(A):
    """
    INPUT:
    A: LongTensor (n,n)
       Adjacency matrix of a graph with n vertices and self loop
    
    OUTPUT:
    Partition: LongTensor of shape (num_connected_components , n): 
               The first row is the indicator of the first connected component
               The second row is the indicator of the second connected component
               etc...
               
    """
    
    B = A
    B_prev = torch.zeros(A.shape,dtype = A.dtype)
    while not torch.equal(B,B_prev):
        B_prev = B
        B = torch.mm(B,A)
        B = (B>0).long()
    Partition = torch.unique(B, dim=0)
    return Partition


def get_mu_nu_from_adacency(A):
    
    """
    This function count the number of connected components of each possible size 
    with the exception of connected component of size 1 (isolated vertices)
    
    For example
    
    mu = [2 ,5 , 10]
    nu = [7, 3,  1]
    
    means that there are 7 connected components of size 2
                         3 connected components of size 3
                         1 connected component of size 15
                         
    and there might be some extra isolated vertices.
    The entries in the vector mu are all distinct.
    
    
    INPUT:
    A: LongTensor (n,n)
       Adjacency matrix of a graph with n vertices and self loop
    
    OUTPUT:
    mu: LongTensor of shape (m,)
    nu: LongTensor of shape (m,)
               
    """
    
    Partition = get_connected_components(A)
    sz_of_connected_comp = torch.sum(Partition,dim=1)
    
    # hist[s] gives the number of connected components of size s
    hist = integers_histogram(sz_of_connected_comp)
    
    # We are not interested in the number of isolated vertices (connected component of size 1)
    # So we remove this entry from the dictionary if it is there
    if 1 in hist:
        del hist[1]
    
    # mu is a LongTensor containing the possible sizes for the connected components
    # nu is a LongTensor 
    mu = torch.tensor( list(hist.keys()) )
    nu = torch.tensor( list(hist.values()) )
    
    return mu, nu  


def get_mu_nu(x,y):
    """
    x,y: LongTensors of shape (L,). These are data points
    This function compute mu and nu for these two data points.
    """
    
    A = create_adjacency(x,y)
    mu, nu = get_mu_nu_from_adacency(A)
    return mu, nu

