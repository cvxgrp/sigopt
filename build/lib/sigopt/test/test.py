'''
Unit tests for sigmoidal programming problems.
'''

# Copyright 2013 M. Udell
# 
# This file is part of SIGOPT version 0.1.0.
# 
# SIGOPT is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# SIGOPT is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with SIGOPT.  If not, see <http://www.gnu.org/licenses/>.

import sigopt.examples as examples
import sigopt.plot as plot

def exercise(example):
    problem = eval('examples.'+example)
    problem.solve(maxiters=15,verbose=False)
    print 'Solved',example,'to error', \
           problem.bounds[-1][1] - problem.bounds[-1][0], \
           'in',len(problem.bounds),'iterations.'
    plot.plot_convergence(problem)
    plot.plot_best_node(problem)
    return problem

def test():
    problem_names = ['bidding(n=16,type="admit",tol=.01)',
                'bidding(n=16,type="pretty",tol=.01)',
                'bidding(n=16,type="profit",tol=.01)',
                'bidding(n=16,type="winnings",B=.5,tol=.01)',
                'ilp(n=25,m=5)',
                'num(n=16,m=10,s=3)',
                'num(n=16,s=3,opt="ring",func="quadratic")',
                'num(n=16,m=10,s=3,opt="local")'
                ]
    
    # make sure all problems can be compiled and solved
    problems = [exercise(p) for p in problem_names]

    # test a few options
    p = problems[0]
    p.check_z=True
    p.solve(maxiters=5)
        
def test_db():
    '''
    Compares lower bound found by solving a randomized LP with initial lower bound
    The values should differ more when the original lower bound is found 
    with an interior point method, such as using cvxopt,
    than with a simplex method like glpk
    '''
    from sigopt.utilities import get_fapx
    from numpy import random, matrix
    from sigopt import Problem
    problem_names = ['bidding(n=16,type="admit",tol=.01)',
                'bidding(n=16,type="pretty",tol=.01)',
                'bidding(n=16,type="profit",tol=.01)',
                'bidding(n=16,type="winnings",B=.5,tol=.01)',
                'ilp(n=25,m=5)',
                'num(n=16,m=10,s=3)',
                'num(n=16,s=3,opt="ring",func="quadratic")',
                'num(n=16,m=10,s=3,opt="local")'
                ]

    def mult_by(num):
        return lambda x: num*x

    def constant(num):
        return lambda x: num

    ## see how much we improve using db LP
    improvement = []
    for example in problem_names:
        problem = eval('examples.'+example)
        f_orig = problem.fs
        problem.sub_tol = .001
        problem.solve(maxiters=1)
        xhat = problem.x # xhat is a solution to the convexified problem
        f_of_xhat = problem.bounds[0][0] #f_of_xhat is its value according to the original objective
        phat = problem.bounds[0][1] # the optimal value of the convexified problem
    
        ## Constraints defining optimal set for convexified problem
        n = len(problem.l)
        slopes, offsets, fapxs = get_fapx(problem.best_node.tight, problem.fs, problem.l, problem.u)
        # first n coordinates correspond to x; second n correspond to f_i^*
        # Cx == d means \sum f_i^* == phat
        C = [[0]*n + [1]*n]
        d = [phat]

        A = []; b = []; 
        for i,(slope_list,offset_list) in enumerate(zip(slopes,offsets)):
            for s,o in zip(slope_list,offset_list):
                # Ax <= b means f[i] - s*x[i] <= o so f[i] <= s*x[i] + o
                A.append([0]*2*n)
                A[-1][i] = -s
                A[-1][i+n] = 1
                b.append(o)
        
        ## Add constraints from original problem
        if problem.constr['A'] is not None and problem.constr['b'] is not None:
            A = A + [row + [0]*n for row in matrix(problem.constr['A']).tolist()]
            b = b + [row[0] for row in matrix(problem.constr['b']).tolist()]
   
        if problem.constr['C'] is not None and problem.constr['d'] is not None:
            C = C + [row + [0]*n for row in matrix(problem.constr['C']).tolist()]
            d = d + [row[0] for row in matrix(problem.constr['d']).tolist()]
        
        big = 100000*max(1,phat) # bounds |f^*_i| - nuisance parameter, shouldn't matter if sufficiently large
        l = problem.l + [-big]*n
        u = problem.u + [big]*n

        ## solve a randomized LP to find a better solution (lower bound)
        w = random.randn(n)
        # first n coordinates correspond to x; second n correspond to f_i^*
        fprime_const = w.tolist() + [0]*n
        f = map(mult_by,fprime_const)
        fprime = map(constant,fprime_const)
        fs = zip(f,fprime)

        random_problem = Problem(l, u, fs, matrix(A), b, matrix(C), d)
        random_problem.solve()
        x_tilde = random_problem.x[:n]
        
        new_LB = sum(map(lambda (fi,xi): fi[0](xi), zip(f_orig,x_tilde)))
    
        # Find true solution pstar & improvement
        problem.solve(maxiters = 100)
        pstar = problem.bounds[-1][1]
        if pstar - f_of_xhat > 0:
            improvement_ratio = (new_LB - f_of_xhat)/(pstar - f_of_xhat)
        elif pstar - new_LB == 0:
            improvement_ratio = 0
        else:
            improvement_ratio = 0
        print example, pstar, new_LB, f_of_xhat, improvement_ratio
        improvement.append(improvement_ratio)
        
    print 'improvements:',improvement
        
def test_rand_refine(solver='glpk',rand_refine = 3,maxiters=10):
    from sigopt import Problem

    problem_names = ['ilp(n=25,m=5)',
                     'ilp(n=50,m=10)',
                     'ilp(n=75,m=15)',
                     'ilp(n=100,m=20)',
                    ]

    ## see how much we improve using db LP
    for example in problem_names:
    
        problem = eval('examples.'+example)    
        problem.solve(maxiters=maxiters,solver=solver)
        before_rand = problem.best_node.LB
        
        new_problem = problem.new()    
        new_problem.solve(maxiters=maxiters,solver=solver,rand_refine=rand_refine)
        after_rand = new_problem.best_node.LB
        print 'improvements:',before_rand, after_rand
        
if __name__=='__main__':
    test_rand_refine()