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
        
def test_rand_refine(solver='cvxpy',rand_refine = 1,maxiters=1):
    '''
    Compares lower bound found by solving a randomized LP with initial lower bound
    The values should differ more when the original lower bound is found 
    with an interior point method, such as using cvxopt,
    than with a simplex method like glpk
    '''
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
        
        new_problem.solve(maxiters = 50, solver = 'glpk', rand_refine = 3)
        pstar = new_problem.LB
        print 'improvements:',before_rand, after_rand, pstar
        
if __name__=='__main__':
    test()
    test_rand_refine()