import math


def solve_int_ineq(a,b):
    """
    Find all the integer vectors x = [x_0, x_1, ..., x_{n-1}] that satisfy
    
    x_i >= 0 for i = 0,...,n-1                           (1)
    a_0 x_0 + a_1 x_1 + ... + a_{n-1} x_{n-1} <= b       (2)
    
    INPUT: 
    a: list of length n contaings ints or floats >0
    b: int or float >= 0
    
    OUTPUT:
    sln: list length num_sln where num_sln is the number of solutions of (1)-(2)
         Each element of the is a tuple (x,s) where s = <a,x>
    """
    n = len(a)
    imax = math.floor(b/a[0])
    E = [ ( [i] , i*a[0] ) for i in range(imax+1)]
    for m in range(1,n):
        Enew = []
        for x,s in E:
            imax = math.floor((b-s)/a[m])
            Enew = Enew + [ ( x+[i] , s+a[m]*i ) for i in range(imax+1) ]   
        E = Enew    
    return E


def weighted_simplex(a,s):
    """
    Given coefficients a = [a_1, ..., a_{n-1}]
    find all the integer vectors x = [x_0, x_1, ..., x_{n-1}] that satisfy
    
    x_i >= 0 for i = 0,...,n-1          
    x_0 + a_1*x_1 + a_2*x_2 + ... + a_{n-1}*x_{n-1} = s    
    
    Note that there are only n-1 coefficient! But x is in R^n! 
    x_0 is always assumed to have a coefficient 1 in front.
    
    INPUT:  a: list of length n-1 containg ints or floats > 0
            s: int
    OUTPUT: simplex: list of length num_sln 
                     Each element of the list is a list of length n
    """
    simplex = []    
    sln = solve_int_ineq(a,s)           # solve a_1*x_1 + ... + a_{n-1} * x_{n-1} <= s
    for x_partial, s_partial in sln:
        x = [s-s_partial] + x_partial   # compute x_0                         
        simplex.append(x) 
    return simplex
    
    
def simplex(n,s):
    """
    Find all the integer vectors x = [x_0, x_1, ..., x_{n-1}] that satisfy
    
    x_i >= 0 for i = 0,...,n-1          
    x_0 + x_1 + ... + x_{n-1} = s       
    
    INPUT:  n,s: int
    OUTPUT: simplex: list length num_sln 
                     Each element of the list is a list of length n
    """
    return weighted_simplex([1]*(n-1), s)

