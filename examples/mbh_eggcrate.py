from __future__ import print_function, absolute_import

import numpy as np
import argparse
from pyoptsparse import Optimization, OPT


# Solving the Egg Crate Function:
# http://benchmarkfcns.xyz/benchmarkfcns/eggcratefcn.html


parser = argparse.ArgumentParser()
parser.add_argument("--opt",help="optimizer",type=str, default='mbh')
args = parser.parse_args()
optOptions = {'alpha': 0.3, 'verbose': True, 'maxTime': 10, 'maxIter': 20}



def objfunc(xdict):
    x = xdict['x']
    y = xdict['y']
    funcs = dict()
    funcs['obj'] = x**2 + y**2 + 25*(np.sin(x)**2 + np.sin(y)**2)
    fail = False
    return funcs, fail


def sens(xdict, funcs):
    x = xdict['x']
    y = xdict['y']
    funcsSens = dict()
    funcsSens['obj'] = {
        'x': np.array([2*(x + 25* np.sin(x) * np.cos(x))]),
        'y': np.array([2*(y + 25* np.sin(y) * np.cos(y))]),

    }
    fail = False
    return funcsSens, fail


# Optimization Object
optProb = Optimization('Lee Problem', objfunc)

# Design Variables
optProb.addVar('x', 'c', value=-5, lower=-5, upper=5)
optProb.addVar('y', 'c', value=-5, lower=-5, upper=5)

# Objective
optProb.addObj('obj')

# Check optimization problem:
print(optProb)

# Optimizer
opt = OPT(args.opt, options=optOptions)

# Solution
sol = opt(optProb, sens=sens)

# Check Solution
print(sol)
