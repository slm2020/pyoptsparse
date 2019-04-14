from __future__ import print_function, absolute_import

import numpy as np
import argparse
from pyoptsparse import Optimization, OPT

# Solving the Gramacy & Lee Function:
# http://benchmarkfcns.xyz/benchmarkfcns/gramacyleefcn.html

parser = argparse.ArgumentParser()
parser.add_argument("--opt",help="optimizer",type=str, default='mbh')
args = parser.parse_args()
optOptions = {'alpha': 2.0, 'verbose': True, 'maxTime': 1, 'maxIter': 3}

def objfunc(xdict):
    x = xdict['xvars']
    funcs = {}
    funcs['obj'] = x[0] * x[3] * (x[0] + x[1] + x[2]) + x[2]
    funcs['con1'] = x[0] * x[1] * x[2] * x[3]
    funcs['con2'] = x[0]*x[0] + x[1]*x[1] + x[2]*x[2] + x[3]*x[3]
    fail = False
    return funcs, fail

def sens(xdict, funcs):
    x = xdict['xvars']
    funcsSens = {}
    funcsSens['obj'] = {'xvars': np.array(
            [x[0] * x[3] + x[3] * (x[0] + x[1] + x[2]) ,
             x[0] * x[3],
             x[0] * x[3] + 1.0,
             x[0] * (x[0] + x[1] + x[2])
             ])}
    jac = [[x[1]*x[2]*x[3], x[0]*x[2]*x[3], x[0]*x[1]*x[3], x[0]*x[1]*x[2]]]
    funcsSens['con1'] = {'xvars': jac}
    jac = [[2.0*x[0], 2.0*x[1], 2.0*x[2], 2.0*x[3]]]
    funcsSens['con2'] = {'xvars': jac}
    fail = False
    return funcsSens, fail

# Optimization Object
optProb = Optimization('HS071 Constraint Problem', objfunc)

# Design Variables
x0 = [5.0, 5.0, 5.0, 5.0]
optProb.addVarGroup('xvars', 4, lower=1, upper=5, value=x0)

# Constraints
# optProb.addCon('con1', lower=25, upper=1e19)
optProb.addCon('con1', lower=25)
# optProb.addCon('con2', lower=40, upper=40)
optProb.addCon('con2', lower=40, upper=40)

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
