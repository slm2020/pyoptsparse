# /bin/env python
"""
pyMBH - A pyOptSparse interface to Monotonic Basin Hopping
work with sparse optimization problems.

Copyright (c) 2019-2020 by Steven L. McCarty
All rights reserved.

Developers:
-----------
- Steven L. McCarty (SLM)
- Robert D. Falck (RDF)

History
-------
    v. 0.1    - Initial Wrapper Creation
"""
from __future__ import absolute_import
from __future__ import print_function
# =============================================================================
# Standard Python modules
# =============================================================================
import time
from copy import deepcopy

# =============================================================================
# External Python modules
# =============================================================================
import numpy as np
import random
from six import iteritems
# ===========================================================================
# Extension modules
# ===========================================================================
from ..pyOpt_optimizer import Optimizer, OPT
from ..pyOpt_optimization import Optimization
from ..pyOpt_error import Error


# =============================================================================
# ALPSO Optimizer Class
# =============================================================================
class MBH(Optimizer):
    """
    MBH Optimizer Class - Inherited from Optimizer Abstract Class

    *Keyword arguments:**

    """

    def __init__(self, *args, **kwargs):

        category = 'Global Optimizer'
        defOpts = {
            'alpha': [float, 2.0],              # Pareto Distribution Alpha Parameter to Determine Hop Distribution
            'maxIter': [int, 1e12],             # Maximum Number of Trials
            'maxTime': [int, 3600],             # Maximum Run Time (s)
            'optimizer': [Optimizer, None],     # Optimizer to be Used

        }

        informs = {
            10: 'iteration limit reached with feasible solution',
            11: 'iteration limit reached without feasible solution',
            20: 'time limit reached with feasible solution',
            21: 'time limit reached without feasible solution',
        }
        super(MBH, self).__init__('MBH', category, defOpts, informs, *args, **kwargs)

    def __call__(self, optProb, sens=None, storeHistory=None, **kwargs):
        """
        This is the main routine used to solve the optimization
        problem.

        Parameters
        ----------
        optProb : Optimization or Solution class instance
            This is the complete description of the optimization problem
            to be solved by the optimizer

        storeHistory : str
            File name of the history file into which the history of
            this optimization will be stored

        Notes
        -----
        The kwargs are there such that the sens= argument can be
        supplied (but ignored here in alpso)
        """
        # ======================================================================
        # ALPSO - Objective/Constraint Values Function
        # ======================================================================
        # def objconfunc(x):
        #     fobj, fcon, fail = self._masterFunc(x, ['fobj', 'fcon'])
        #     return fobj, fcon

        # Save the optimization problem and finalize constraint
        # jacobian, in general can only do on root proc
        self.optProb = optProb
        self.optProb.finalizeDesignVariables()
        self.optProb.finalizeConstraints()
        self._setInitialCacheValues()

        # if len(optProb.constraints) == 0:
        #     self.unconstrained = True
        #
        # xl, xu, xs = self._assembleContinuousVariables()
        # xs = numpy.maximum(xs, xl)
        # xs = numpy.minimum(xs, xu)
        # n = len(xs)
        # ff = self._assembleObjective()
        # types = [0] * len(xs)
        # oneSided = True
        # if self.unconstrained:
        #     m = 0
        #     me = 0
        # else:
        #     indices, blc, buc, fact = self.optProb.getOrdering(
        #         ['ne', 'le', 'ni', 'li'], oneSided=oneSided, noEquality=False)
        #     m = len(indices)
        #
        #     self.optProb.jacIndices = indices
        #     self.optProb.fact = fact
        #     self.optProb.offset = buc
        #     indices, __, __, __ = self.optProb.getOrdering(
        #         ['ne', 'le'], oneSided=oneSided, noEquality=False)
        #     me = len(indices)
        #
        # nobj = 1


        if self.optProb.comm.rank == 0:

            # Set history/hotstart/coldstart
            self._setHistory(storeHistory, None)

            # Setup argument list values
            options = self.getOption

            # dyniI = self.getOption('dynInnerIter')
            # if dyniI == 0:
            #     self.setOption('minInnerIter', opt('maxInnerIter'))
            #
            # if not opt('stopCriteria') in [0, 1]:
            #     raise Error('Incorrect Stopping Criteria Setting')
            #
            # if opt('fileout') not in [0, 1, 2, 3]:
            #     raise Error('Incorrect fileout Setting')
            #
            # if opt('seed') == 0:
            #     self.setOption('seed', time.time())

            # As far as I can tell, there is no need for this bulk attribute.
            # ALPSO calls the objconfunc iteratively for each particle in the
            # swarm, so we can deal with them one at a time, just as the other
            # optimizers.
            # self.optProb.bulk = opt('SwarmSize')

            # Run ALPSO
            t0 = time.time()

            #optProbRun = Optimization(optProb.name, optProb.objFun)
            best_objective = 1e20
            best_sinf = 1e20
            best_x = None
            solution_found = False
            while time.time() - t0 < 1:

                # perturb inputs
                rand = np.random.pareto(0.25) * np.random.choice([-1,1])
                print('Pareto Random Number:', rand)
                print('Best Objective:', best_objective)
                blx, bux, xs = self._assembleContinuousVariables()

                for name, var_list in iteritems(optProb.variables):
                    #print(name,var_list)
                    #print(optProb.variables[name][0].value)
                    val = optProb.variables[name][0].value
                    new_val = val + (rand/100) * val
                    optProb.variables[name][0].value = new_val
                    print("val", val, "new_val", new_val)

                # run problem
                opt = OPT('snopt')
                sol = opt(optProb, sens=sens)

                # get solution
                sub_objective = sol.fStar
                sub_inform = sol.optInform['value'][0]
                sub_sinf = sol.sInf[0]
                sub_ninf = sol.nInf
                sub_xStar = sol.xStar

                print("sub_obj", sub_objective)
                print("sub_inform", sub_inform)
                print("sub_sinf", sub_sinf)
                print("sub_xstar", sub_xStar)


                if sub_inform == 1 and sub_objective < best_objective:

                    solution_found = True
                    best_objective = sub_objective
                    print("New Best Objective Found:", best_objective)
                    best_xStar = sub_xStar

                    for name, var_list in iteritems(optProb.variables):
                        optProb.variables[name][0].value = best_xStar[name]



                elif sub_inform != 1 and sub_sinf < best_sinf and not solution_found:

                    best_sinf = sub_sinf
                    print("More Feasible Solution Found:", best_objective)
                    best_xStar = sub_xStar

                    for name, var_list in iteritems(optProb.variables):
                        optProb.variables[name][0].value = best_xStar[name]


                else:
                    # reset
                    for name, var_list in iteritems(optProb.variables):
                        optProb.variables[name][0].value = val




            # opt_x, opt_f, opt_g, opt_lambda, nfevals, rseed = self.alpso.alpso(
            #     n, m, me, types, xs, xl, xu, opt('SwarmSize'), opt('HoodSize'),
            #     opt('HoodModel'), opt('maxOuterIter'), opt('maxInnerIter'),
            #     opt('minInnerIter'), opt('stopCriteria'), opt('stopIters'),
            #     opt('etol'), opt('itol'), opt('rtol'), opt('atol'), opt('dtol'),
            #     opt('printOuterIters'), opt('printInnerIters'), opt('rinit'),
            #     opt('vinit'), opt('vmax'), opt('c1'), opt('c2'), opt('w1'),
            #     opt('w2'), opt('ns'), opt('nf'), opt('vcrazy'), opt('fileout'),
            #     opt('filename'), None, None, opt('seed'),
            #     opt('Scaling'), opt('HoodSelf'), objconfunc)
            optTime = time.time() - t0

            # Broadcast a -1 to indcate NSGA2 has finished
            self.optProb.comm.bcast(-1, root=0)

            # Store Results
            sol_inform = {}
            # sol_inform['value'] = inform
            # sol_inform['text'] = self.informs[inform[0]]

            # Create the optimization solution
            # sol = self._createSolution(optTime, sol_inform, opt_f, opt_x)
            # for key in sol.objectives.keys():
            #     sol.objectives[key].value = opt_f
        else:  # We are not on the root process so go into waiting loop:
            self._waitLoop()
            sol = None

        # Communication solution and return
        sol = self._communicateSolution(sol)

        return sol

    def _on_setOption(self, name, value):
        if name == 'parallelType':
            value = value.upper()
            if value == 'EXT':
                try:
                    from . import alpso_ext
                    self.alpso = alpso_ext
                except ImportError:
                    raise ImportError('pyALPSO: ALPSO EXT shared library failed to import.')

            else:
                raise ValueError("parallel_type must be either '' or 'EXT'.")

    def _on_getOption(self, name, value):
        pass

    def _communicateSolution(self, sol):
        if sol is not None:
            sol.userObjCalls = self.optProb.comm.allreduce(sol.userObjCalls)
            sol.comm = None
        sol = self.optProb.comm.bcast(sol)
        sol.objFun = self.optProb.objFun
        sol.comm = self.optProb.comm

        return sol
