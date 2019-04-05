from __future__ import print_function, absolute_import

import numpy as np
import argparse
from pyoptsparse import Optimization, OPT

parser = argparse.ArgumentParser()
parser.add_argument("--opt",help="optimizer",type=str, default='mbh')
args = parser.parse_args()
optOptions = {}

# def objfunc(xdict):
#     x = xdict['xvars']
#     funcs = dict()
#     funcs['obj'] = x[0]**2 + x[1]**2 + 25*(np.sin(x[0])**2 + np.sin(x[1])**2)
#     fail = False
#     return funcs, fail
#
#
# def sens(xdict, funcs):
#     x = xdict['xvars']
#     funcsSens = dict()
#     funcsSens['obj'] = {'xvars': np.array(
#             [2*x[0] + 25*(np.sin(x[0])**2 + np.sin(x[1])**2)*2*np.sin(x[0])*np.cos(x[0]) ,
#              2 * x[1] - 25 * (np.sin(x[0]) ** 2 + np.sin(x[1]) ** 2) * 2 * np.sin(x[1])*np.sin(x[1])
#              ])}
#     fail = False
#     return funcsSens, fail

def objfunc(xdict):
    x = xdict['x']
    funcs = dict()
    funcs['obj'] = np.sin(10*np.pi*x)/(2*x) + (x-1)**4
    fail = False
    return funcs, fail


def sens(xdict, funcs):
    x = xdict['x']
    funcsSens = dict()
    funcsSens['obj'] = {'x': np.array(
            [
                -np.sin(10*np.pi*x)/(2*x**2) + 4*(x-1)**3 + 5*np.pi*np.cos(10*np.pi*x)/x
            ])}
    fail = False
    return funcsSens, fail


# Optimization Object
optProb = Optimization('Lee Problem', objfunc)

# Design Variables
optProb.addVar('x', 'c', value=0.54856, lower=-0.5, upper=2.5)

# Objective
optProb.addObj('obj')

# Check optimization problem:
#print(optProb)

# Optimizer
opt = OPT(args.opt, options=optOptions)

# Solution
sol = opt(optProb, sens=sens)

# Check Solution
print(sol)
