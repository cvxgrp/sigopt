'''
Utilities for constructing and maximizing concave envelopes
to solve sigmoidal programming problems using branch and bound
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
# along with SIGOPT.  If not, see <http://www.gnu.org/licenses/>.from __future__ import division

import math, numpy
import solvers
import cvxopt, cvxopt.solvers
# cvxopt solver settings
cvxopt.solvers.options['show_progress'] = False
cvxopt.solvers.options['refinement'] = 2
cvxopt.solvers.options['abstol'] = 1e-16
cvxopt.solvers.options['reltol'] = 1e-13
## glpk solver settings
# quiet
cvxopt.solvers.options['LPX_K_MSGLEV'] = 0
from cvxopt.base import matrix, exp
from cvxopt.modeling import variable, op, max, sum

import sys
from operator import mul
from scipy.optimize import brentq

# set numerical tolerance 
TOL = 1e-16

def find_root( fun, a, b, tol = TOL ):
    '''
    Finds a zero of function fun that lies between a and b.
    '''
    return brentq( fun, float(a), float(b))
    
def find_z(ftup,l,u,tol=.01,verbose=False):
    '''
    search for boundary zi between convex and concave regions
    f should be ctsly diffable and strictly convex or concave except at z 
    for this to work
    '''
    f,fprime = ftup
    def vexity(z,n=1000):  
        dz = (u-l)/n
        return pow(n,2)*(f(z+dz) + f(z-dz) - 2*f(z))
    if vexity(l) < 0:
        z = l
    elif vexity(u) > 0:
        z = u
    else:
        z = find_root( vexity, l, u)
    if verbose: print 'found l,z,u:',l,z,u
    return (f,fprime,z)
  
def check_z(p,n=None,fapx=None,y=None):
    '''
    checks that the third entry of the tuple for each f 
    does in fact allow a concave envelope for the function can be correctly constructed
    over the region [p.l,p.u]
    '''
    if not n:
        n=p
    if not fapx or not y:
        tight = [find_concave_hull(li,ui,fi,[],p.sub_tol) for li,ui,fi in zip(p.l,p.u,p.fs)]
        fapx = get_fapx(tight, p.fs, p.l, p.u, y=y, tol = p.sub_tol)[-1]

    for i in range(len(n.l)):
        x = list(numpy.linspace(n.l[i],n.u[i],500))
        for xi in x:
            error = fapx[i](xi) - p.fs[i][0](xi)
            if error < - p.tol/len(n.l):
                raise ValueError('Could not construct a concave envelope for function %d.\
                                  Please check the value of z provided.'%i)
  
def find_concave_hull( li, ui, fi, tight=None, tol=.01 ):
  '''
  Returns a list of points at which the linear approximation to fi should be tight
  in order to form a close convex hull of the function fi on the interval 
  between the scalars li and ui
  
  fi should be convex for x < zi and concave for x > zi
  fprime can be any function such that \int_l^x fprime(x) dx = f(x)-f(l)
  '''
  f = fi[0]; fprime = fi[1]; zi = fi[2]
    
  # If fi is convex from li to ui (ui<zi)
  # or the tangent curve to fi intersecting li touches fi at a point z>ui (f'(u)>(f(u)-f(l))/(u-l)),   
  # then the concave hull is just the line connecting (li,f(li)) with (ui,f(ui))
  if li==ui or ui <= zi or fprime(ui) > (f(ui)-f(li))/(ui-li):
    return []
  
  # Find the farthest-left point wi for which fconv(wi) = f(wi)
  if li >= zi:
    wi = li
  else:
  	# f is assumed to have continuous derivative
  	# so we know we can find z st f'(z) = (f(z)-f(l))/(z-l) because 
  	# f'(z) >= (f(z)-f(l))/(z-l) for z = 0 by convexity of f for z<=w, and 
  	# f'(z) <= (f(z)-f(l))/(z-l) for z = u because we checked above
    fl = f(li)
    wi = find_root( lambda wi: fprime(wi)*(wi-li) - (f(wi)-fl), zi, ui )
  # Now find a sufficiently tight approximation everywhere between z and ui
  if tight: tight_old = tight
  else: tight_old = []
  tight = [wi,ui]
  tight = tighten(tight,tol,f,fprime)
  for p in tight_old:
    if p > wi and p < ui:
      tight.append(p)
  return matrix(sorted(set(tight)))

def tighten(tight,tol,f,fprime,verbose=False):
  '''Finds a set of points to use in constructing a good concave envelope for f
  
  Returns a set of points tight such that the approximant to f 
  defined by the tangents at those points is within tol of f
  '''
  # The approximation is worst at the points where the tangents intersect
  # Test points p are therefore where the tangents intersect
  more_tight=[tight[0]]
  modified_flag = False
  for i in range(len(tight) - 1):
    a = tight[i]; b = tight[i+1]
    
    # if tangents don't intersect between a and b, and a and b are different, 
    # then f must be flat (or nearly) between a and b if it is concave.
    w = (f(b)-f(a))/(b-a) # average slope b/w a and b
    if (fprime(b)-w)*(fprime(a)-w)>=0:
    	# do a sanity check to make sure f is (probably) concave between a and b
    	if f((b+a)/2) < (f(a) + f(b))/2:
    		raise ValueError('the function f = %s is not concave between %f and %f\n since f((b+a)/2) = %.5f and (f(a) + f(b))/2 = %.5f' % (f,a,b,f((b+a)/2),(f(a) + f(b))/2))

	# Find point where tangents intersect
    else:
		p = (f(a) - fprime(a)*a - f(b) + fprime(b)*b) / (fprime(b) - fprime(a))
		# add it unless difference between linear appx at a to f(p) is within tolerance
		if f(a) + fprime(a)*(p-a) - f(p) > tol:
		  if verbose: print 'adding point',p,'where tangents intersect'
		  more_tight.append(p)
		  modified_flag = True
    more_tight.append(b)
  if modified_flag:
	return tighten(more_tight,tol,f,fprime)
  else:
	return tight
  
def get_fapx(tight, fs, l, u, y = None, tol = 1e-10, debug = False ):
    '''
    Returns the piecewise linear approximation fapxi to fi for each fi in fs 
    that is tight at each point in tight[i], as well as at l and u, 
    in 3 formats: 

      slopes/offsets: lists of slopes and offsets of linear functionals approximating f.             
                      fapxi(x) = max_j slopes[i,j]*x + offsets[i,j]
                      These obey
    
               for i,slope_list,offset_list in enumerate(zip(slopes,offsets)):
                  for s,o in zip(slope_list,offset_list):
                      fapxs[i] <= s*x[i] + o
                    
      fapxs: a list of functions fapxi
  
      ms: as cvxopt modeling function (multiplied by -1, 
          because cvxopt minimizes, and doesn't maximize)
          (only computed if cvxopt.modeling.variable y is given)
        
    Also returns 
    '''  
    if y is not None: ms = []
    slopes = []
    offsets = []
    fapxs = []
  
    for i in range(len(l)):
        # cast to float, since cvxopt breaks if numpy.float64 floats are used
        f = lambda x: float(fs[i][0](x))
        fprime = lambda x: float(fs[i][1](x))
        # If there are any points at which the apx should be tight, 
        # make it tight there
        if tight[i]:
            if tight[i][0]>l[i]:
                if y is not None: 
                    m = max([-(f(l[i])+\
                        (f(tight[i][0])-f(l[i]))/(tight[i][0]-l[i])*(y[i]-l[i]))] + 
                        [-float(f(z)) - float(fprime(z))*(y[i]-z) for z in tight[i]])
                slope = [(f(tight[i][0]) - f(l[i]))/(tight[i][0]-l[i])] + \
                        [ float(fprime(z)) for z in tight[i]]
                offset = [f(l[i])+(f(tight[i][0])-f(l[i]))/(tight[i][0]-l[i])*(-l[i])] + \
                         [float(f(z)) - float(fprime(z))*z for z in tight[i]]
            else:
                if y is not None: 
                    m = max([-float(f(z)) - float(fprime(z))*(y[i]-z) for z in tight[i]])
                slope = [float(fprime(z)) for z in tight[i]]
                offset = [float(f(z)) - float(fprime(z))*z for z in tight[i]]
        # check if upper and lower bounds are the same
        elif (u[i] - l[i] < tol):
            if y is not None: 
                    m = f(l[i]) + 0*y[i]
            slope = [0]
            offset = [f(l[i])]
        # otherwise we only use the line connecting (l,f(l)) with (u,f(u))
        else:
            if y is not None: 
                    m = -(f(l[i]) + (f(u[i]) - f(l[i]))/(u[i]-l[i]) * (y[i]-l[i]))
            slope = [(f(u[i]) - f(l[i]))/(u[i]-l[i])]
            offset = [f(l[i]) + (f(u[i]) - f(l[i]))/(u[i]-l[i]) * (-l[i])]
    
        if y is not None: 
                    ms.append(m); 
        slopes.append(slope); offsets.append(offset)
    
        def fapxi(xi,slope=slope,offset=offset):
            fi = []
            for s,o in zip(slope,offset):
                fi.append(s*xi + o)
            return min(fi)
        
        fapxs.append(fapxi)
   
    if not y is None:
        return (slopes,offsets,fapxs)
    else:
        return (slopes,offsets,fapxs)
  
def compute_tight(node,problem):
    '''
    Find the points at which the approximation fapx should be tight
    '''
    node.tight = [find_concave_hull(node.l[i],node.u[i],problem.fs[i],
                                    tight=[],
                                    tol=problem.sub_tol) 
                  for i in range(len(node.l))]

def compute_ULB(node,problem):
    '''
    Finds an upper and lower bound for maximum objective function value 
    over the region 
                    node.l<=x<=node.u
                    A*x <= b
                    C*x == d
    '''
    compute_tight(node,problem)
    ULB = maximize_fapx(node,problem)
    if not ULB:
      node.UB = -float('inf'); node.LB = -float('inf')
      node.x = None;  node.maxdiff_index = None 
    else:
      node.UB = ULB['fapx']
      node.LB = ULB['f']
      node.x = ULB['x']
      node.maxdiff_index = ULB['maxdiff_index']
        
def maximize_fapx( node, problem):
    '''
    Finds the value of y solving maximize sum(fapx(x)) subject to:
                       constr
                       l <= x <= u
    fapx is calculated by approximating the functions given in fs
    as the concave envelope of the function that is tight, 
    for each coordinate i, at the points in tight[i]
    fs should be a list of tuples containing the function and its derivative
    corresponding to each coordinate of the vector x.
  
    Returns a dict containing the optimal variable y as a list, 
    the optimal value of the approximate objective,
    and the value of sum(f(x)) at x.
    '''
    if problem.solver == 'glpk':
        return solvers.maximize_fapx_glpk( node, problem )
    elif problem.solver == 'cvxopt':
        return solvers.maximize_fapx_cvxopt( node, problem )
    elif problem.solver == 'cvxpy':
        import cvxpy
        return solvers.maximize_fapx_cvxpy( node, problem )
    else:
        raise ValueError('Unknown solver %s'%solver)
	
def format_constraints_cvxopt(problem):
        variable = cvxopt.modeling.variable(n,'x')
        problem.cvxopt_constr = []
        
        if A is not None and b is not None:
            A = cvxopt.matrix(A,tc='d')
            b = cvxopt.matrix(b,tc='d')
            ineq = (A*variable <= b)
            problem.cvxopt_constr.append(ineq)
        if C is not None and d is not None:
            C = cvxopt.matrix(C,tc='d')
            d = cvxopt.matrix(d,tc='d')
            eq = (C*variable == d)
            problem.cvxopt_constr.append(eq)
        problem.cvxopt_variable = variable
	
def format_constraints(problem,A,b,C,d,constr=None,variable=None,n=None,format='sigopt'):
    '''
    Format constraints for problem
    '''
    if A is not None and b is not None:
        A = cvxopt.matrix(A)
        b = cvxopt.matrix(b)
    if C is not None and d is not None:
        C = cvxopt.matrix(C)
        d = cvxopt.matrix(d)
    constr = {'A':A,'b':b,'C':C,'d':d}
        
    # make sure dimensions of inputs match   
    n = len(problem.l)
    if A is not None and b is not None:
        if not A.size[0] == b.size[0]:
            raise ValueError('Check dimensions of A and b')
        if not A.size[1] == n:
            raise ValueError('Check dimensions of A')
    if C is not None and d is not None:
        if not C.size[0] == d.size[0]:
            raise ValueError('Check dimensions of C and d')
        if not C.size[1] == n:
            raise ValueError('Check dimensions of C')
            
    return constr
            
def scoop(p,node,slopes,offsets):
    '''
    Save data in general format for scoop modeling library consumption
    '''
    fn = 'scoop/%s.n-%d.pickle' % (problem.name,len(node.l))
    print 'writing problem data to',fn
    with open(fn,'w') as f:
        pickle.dump({'l':node.l,'u':node.u,
                     'A':problem.A,'b':problem.b,
                     'slopes':slopes,'offsets':offsets},f)