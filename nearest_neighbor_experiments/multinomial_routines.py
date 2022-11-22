import math
import scipy.special

def multinomial(lst):
    """
    INPUT: list or tuple of integers
    OUTPUT: multinomial value of this list
    """
    sum_of_k = lst[0]
    p = 1
    for k in lst[1:]:
        sum_of_k += k
        p = p * scipy.special.comb(sum_of_k, k, exact=True)
    return p

def contains_negative_integer(lst):
    """
    return true is the list contains an integer <0
    """
    return any(i < 0 for i in lst)

def multinomial_star(lst):
    """
    Like multinomial, except that it returns 0
    if the list contains an integer <0
    """
    if contains_negative_integer(lst):
        return 0
    else:
        return multinomial(lst)
    

    
