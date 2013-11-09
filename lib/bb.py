#!/usr/bin/env python
'''
Implements Problem and Node container classes
to solve sigmoidal programming problems using branch and bound.
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

import utilities
from operator import concat
from Queue import PriorityQueue, Queue
from parallel import MaxQueue

class Problem(object):
    '''Container for problem parameters
    
    Represents the problem
    
              maximize sum(f_i(x_i)) 
              subject to:
                       A*x <= b
                       Cx == d
                       l <= x <= u
                       constr
                       
    where each fuction f_i is sigmoidal.
    
    Solves the problem using a branch and bound solver to specified accuracy
    
    Arguments:
        l       :  a list of upper bounds on variable x
        u       :  a list of lower bounds on variable x
        fs      :  a list of tuples (f,fprime,z) where fprime gives the derivative of f.
                   f should be convex for x<z and concave for x>z 
        A       :  an arbitrary matrix (n x m)
        b       :  an arbitrary vector (m x 1)
        tol     :  the accuracy required for the problem
        sub_tol :  the accuracy with which to compute the concave envelopes
    
    problem.solve(maxiters) runs the branch and bound solver 
        until a solution of accuracy self.tol is reached, or until
        maxiters concave subproblems have been solved.
    
    problem.best_node is the best node found so far 
        (the one with the highest lower bound) at any time.
        
    problem.LB is the best lower bound on the optimal value found so far; 
        it is achieved by problem.best_node.x
    
    problem.partition is a priority queue containing all nodes under consideration
        indexed by their upper bounds, in decreasing order.
        
    problem.bounds is a list of the bounds obtained after each iteration. 
    '''
    
    def __init__(self,l,u,fs,
                 A=None,b=None,C=None,d=None,constr=None,variable=None,
                 tol=.01,sub_tol=None,name='',nthreads = 1,check_z=False):
        
        # parameters
        self.tol = tol
        if sub_tol:
            self.sub_tol = sub_tol
        else:
            self.sub_tol = tol
        if name: 
            self.name = name
        else:
            self.name = 'sp'
        self.nthreads = nthreads
        
        # box constraints
        # cast everything to float, since cvxopt breaks if numpy.float64 floats are used
        self.l = [float(li) for li in l]
        self.u = [float(ui) for ui in u]
        
        # other constraints         
        self.constr = utilities.format_constraints(self,A,b,C,d,constr)
        
        self.fs = fs
                
        # check dimensions and adequacy of data
        if not (len(fs)==len(l) and len(fs)==len(u)):
            raise ValueError('Check problem dimensions')
        for i,f in enumerate(self.fs):
            if len(f)<3:    
                self.fs[i] = utilities.find_z(f,l[i],u[i],self.sub_tol)
        self.check_z = check_z
            
        # initialize        
        self.x = self.u
        self.best_node = Node(l,u,self)
        self.LB = -float('inf')
        self._bounds = Queue()
        self.bounds = []
        
        # sort nodes for splitting in descending order by upper bound
        self.partition = MaxQueue()
        self.partition.put((float('inf'),self.best_node))
    
    def run_serial(self, maxiters = 0, verbose = False, prune = False, tol = None,
                   solver = 'glpk', rand_refine = 0):
        '''
        Finds a solution of quality problem.tol to the problem.
        
        The optimal node found at any point (ie, the one with the best lower bound 
        and an x that achieves it) is stored in problem.best_node.
        
        The algorithm works by popping the node with the highest upper bound off of the 
        partition, splitting it, and computing bounds for the resulting subrectangles.
        The subrectangles are then inserted into the problem.partition.
        
        The algorithm terminates when the stopping criterion is met, 
        ie when the highest upper bound is less than problem.tol
        greater than the highest lower bound,
        or after maxiters subproblems have been solved.
        
        solver chooses which solver to use to solve convex subproblems. Options
        include cvxpy, cvxopt, and glpk.
        
        if rand_refine > 0, then the lower bound is computed using the best point 
        given by solving rand_refine random LPs, 
        as prescribed by a forthcoming paper on duality bounds. 
        '''
        self.solver = solver
        self.rand_refine = rand_refine
        if solver == 'cvxopt':
            import cvxopt
            from cvxopt.base import matrix, spmatrix, exp
            from cvxopt.modeling import variable, op, max, sum
            utilities.format_constraints_cvxopt(self)
        elif solver == 'cvxpy':
            import cvxpy
        elif solver == 'glpk':
            import glpk

        if tol is None: tol = self.tol
        iter = 0
        while not maxiters or iter < maxiters:
            iter += 1
            UB, node = self.partition.get_nowait()
            
            # initialize bounds if no subproblem has yet been solved
            if UB == float('inf'):
                utilities.compute_ULB(node, self)
                self.partition.put((node.UB,node))
                continue
        
            # record bounds
            self._bounds.put((self.LB,UB))
            if verbose: 
                print 'Bounds',self.LB,UB,'found by',self.name
                
            # stopping criterion
            if UB - self.LB < tol: 
              # a tolerable solution has been found!
              # put current node back in the partition
              if verbose: print 'Stopping criterion reached by %s' % (self.name)
              self.partition.put((UB,node))
              break
            if UB == -float("inf"):
              if verbose: print 'Problem infeasible'
              break
        
            # keep going
            for child in node.split(self):
                if verbose: 
                    print 'split into child with bounds',child.LB,child.UB
                if child.LB > self.LB:
                    if verbose: 
                        print 'Node with best lower bound %.4f found' \
                               % (child.LB)
                    self.LB = child.LB
                    self.best_node = child
                # only put child into partition if child is active
                if prune is False or child.UB >= self.LB:
                    self.partition.put((child.UB,child))
                else:
                    if verbose:
                        print 'Discarded node with bounds',child.LB,child.UB,'since LB is',self.LB
        self._get_bounds()
        self.x = self.best_node.x
        
    def solve(self,*args,**kwargs):
        '''
        Convenience wrapper around run_serial
        '''
        self.run_serial(*args,**kwargs)
        
    def _get_bounds(self):
        while not self._bounds.empty():
            next = self._bounds.get()
            self.bounds.append(next)
        return self.bounds
        
    def new(self):
        '''
        Returns a new problem with the same objective, constraints, and parameters
        '''
        return Problem(self.l,self.u,self.fs,
                       tol=self.tol,sub_tol=self.sub_tol,name=self.name,
                       nthreads = self.nthreads,check_z=self.check_z,
                       **self.constr)
      
class Node(object):
    '''
    A generic node structure to implement branch and bound for a maximization problem
    
    self.l and self.u give the corners of the rectangle that the node represents, 
        ie, Q = (l_1,u_1) x ... x (l_n,u_n)
    
    when the __init__ method is called with an optional problem as argument, 
    the following properties are also computed:
    
        UB is an upper bound on the optimal value over the node
        LB is an lower bound on the optimal value over the node
        x is the point achieving that lower bound   
        tight gives a set of points at which the concave upper approximation of f
            is tight (used for computing piecewise linear concave upper bound).
    '''
    def __init__(self,l,u,problem=None,**kwargs):
        self.l = list(l)
        self.u = list(u)
        if problem and hasattr(problem,'solver'):
            utilities.compute_ULB(self,problem)
        for key in kwargs:
            setattr(self,key,kwargs[key])
    
    def split(self,problem,debug=False):
        '''
        Splits the node into two subnodes
        Splits along the coordinate for which the gap between f and fapx is highest
        with the boundary between regions at the current optimal solution
        
        returns the two subnodes
        '''
        u_left = list(self.u); u_left[self.maxdiff_index] = self.x[self.maxdiff_index]
        left_child = Node(self.l,u_left,problem)
        l_right = list(self.l); l_right[self.maxdiff_index] = self.x[self.maxdiff_index]
        right_child = Node(l_right,self.u,problem)
        return (left_child,right_child)
      
    def __repr__(self):
        return "Node(%r,%r)" % (self.l,self.u)
      
    def __str__(self):
        try:
            return "Node(%r,%r,x=%r,LB=%r,UB=%r)" % (self.l,self.u,self.x,self.LB,self.UB)
        except:
            return "Node(%r,%r)" % (self.l,self.u)