from __future__ import print_function, absolute_import

import numpy as np
import argparse
from pyoptsparse import Optimization, OPT


# Solving the Egg Crate Function:
# http://benchmarkfcns.xyz/benchmarkfcns/eggcratefcn.html
# global min: 0.0 at x,y = 0,0

parser = argparse.ArgumentParser()
parser.add_argument("--opt",help="optimizer",type=str, default='mbh')
args = parser.parse_args()
optOptions = {'alpha': 0.25, 'verbose': True, 'maxTime': 120, 'stallIters': 50}


def objfunc(xdict):
    x = xdict['x']
    y = xdict['y']
    funcs = dict()
    funcs['obj'] = x**2 + y**2 + 25*(np.sin(x)**2 + np.sin(y)**2)
    funcs['con1'] = x

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
    funcsSens['con1'] = {'x': np.array([1])}

    fail = False
    return funcsSens, fail


# Optimization Object
optProb = Optimization('Egg Crate Problem', objfunc)

# Design Variables
optProb.addVar('x', 'c', value=50, lower=-50, upper=50)
optProb.addVar('y', 'c', value=-50, lower=-50, upper=50)

# Objective
optProb.addObj('obj')

# Constraints
# optProb.addCon('con1', lower=25, upper=1e19)
optProb.addCon('con1', upper=-10)

# Check optimization problem:
print(optProb)

# Optimizer
opt = OPT(args.opt, options=optOptions)

# Solution
sol = opt(optProb, sens=sens)

# Check Solution
print(sol)
