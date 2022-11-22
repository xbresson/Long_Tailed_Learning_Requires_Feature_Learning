from eval_formula import get_upperbound
import argparse

parser = argparse.ArgumentParser(description='formula')
parser.add_argument('-L' ,'--L'  , type=int, default=9)
parser.add_argument('-nw','--nw' , type=int, default=150)
parser.add_argument('-nc','--nc' , type=int, default=5)
parser.add_argument('-R' ,'--R'  , type=int, default=1000)
args = parser.parse_args()

sc = args.nw // args.nc
t = 2*args.R-1

print('Inequality From Theorem 7')

for ell in range(args.L,1,-1):
    p = get_upperbound(L=args.L, sc=sc ,nc=args.nc, t=t, ell=ell) 
    print('Choosing ell = {} gives the upper bound:  \t 1 - err(F,psi,T) <=  {:.4f} nstar + 1/R \t  for all F and psi'.format(ell,p))