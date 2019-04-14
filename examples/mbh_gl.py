from __future__ import print_function, absolute_import

import numpy as np
import argparse
from pyoptsparse import Optimization, OPT

# Solving the Gramacy & Lee Function:
# http://benchmarkfcns.xyz/benchmarkfcns/gramacyleefcn.html

parser = argparse.ArgumentParser()
parser.add_argument("--opt",help="optimizer",type=str, default='mbh')
args = parser.parse_args()
optOptions = {'alpha': 0.9, 'verbose': True, 'maxTime': 10, 'stallIters': 10}


def objfuncLee(xdict):
    x = xdict['x']
    funcs = dict()
    funcs['obj'] = np.sin(10*np.pi*x)/(2*x) + (x-1)**4
    fail = False
    return funcs, fail


def sensLee(xdict, funcs):
    x = xdict['x']
    funcsSens = dict()
    funcsSens['obj'] = {'x': np.array(
            [
                -np.sin(10*np.pi*x)/(2*x**2) + 4*(x-1)**3 + 5*np.pi*np.cos(10*np.pi*x)/x
            ])}
    fail = False
    return funcsSens, fail


# Optimization Object
optProb = Optimization('Lee Problem', objfuncLee)

# Design Variables
optProb.addVar('x', 'c', value=2.5, lower=-0.5, upper=2.5)

# Objective
optProb.addObj('obj')

# Check optimization problem:
print(optProb)

# Optimizer
opt = OPT(args.opt, options=optOptions)

# Solution
sol = opt(optProb, sens=sensLee)

# Check Solution
print(sol)
