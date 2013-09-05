'''
LP solvers for sigmoidal programming problems.

Both cvxopt interfaces and py-glpk interfaces are included
py-glpk is preferred since without the bindings to glpk, 
cvxopt sometimes refuses to solve an LP for technical reasons.
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

import glpk, numpy
from cvxopt.base import matrix, spmatrix, exp
from cvxopt.modeling import variable, op, max, sum
import utilities

def cvxopt_modeling2cvxopt_matrices(op):
    '''
    Converts cvxopt.modeling.op instance into its matrices c,A,b,G,h 
    where op represents the problem
    
    minimize c'x
    st       Ax = b
             Gx+h = s
             s >= 0
    '''
    t = op._inmatrixform(format)

    if t is None:
        lp1 = self
    else:
        lp1, vmap, mmap = t[0], t[1], t[2]

    variables = lp1.variables()
    if not variables: 
        raise TypeError('lp must have at least one variable')
    x = variables[0]
    c = lp1.objective._linear._coeff[x]
    #if _isspmatrix(c): c = matrix(c, tc='d')

    inequalities = lp1._inequalities
    if not inequalities:
        raise TypeError('lp must have at least one inequality')
    G = inequalities[0]._f._linear._coeff[x]
    h = -inequalities[0]._f._constant

    equalities = lp1._equalities
    if equalities:
        A = equalities[0]._f._linear._coeff[x]
        b = -equalities[0]._f._constant
    elif format == 'dense':
        A = matrix(0.0, (0,len(x)))
        b = matrix(0.0, (0,1))
    else:
        A = spmatrix(0.0, [], [],  (0,len(x)))
        b = matrix(0.0, (0,1))
    
    return(c,A,b,G,h)
    
def cvxopt_matrices2pyglpk(c,A,b,G,h):
    '''
    Converts optimization problem
    
    minimize c'x
    st       Ax = b
             Gx+h = s
             s >= 0
             
    into glpk lp format
    '''
    m,n = A.size
    k,n = G.size

    lp = glpk.LPX()        # Create empty problem instance
    lp.obj.maximize = False # Set this as a minimization problem
    lp.rows.add(m+k)         # Append m+k rows (constraints) to this instance
    for i,row in enumerate(lp.rows):      # Iterate over all rows
        if i < m:
            row.bounds = b[i], b[i] # Set bound -inf <= pi < inf
        else:
            row.bounds = None, h[i-m]
    lp.cols.add(n)         # Append n columns (variables) to this instance
    for i,col in enumerate(lp.cols):      # Iterate over all columns
        col.bounds = None, None     # Set bound -inf <= xi < inf
    lp.obj[:] = c
    lp.matrix = list(A) + list(G) # p = [A;G] x
    return lp
             
def sigopt2pyglpk(slopes,offsets,l,u,A,b,C,d):
    '''
    Converts optimization problem
    
    maximize sum(f)
    st       Ax <= b
             Cx == d
             l <= x <= u
             for i,slope_list,offset_list in enumerate(zip(slopes,offsets)):
                for s,o in zip(slope_list,offset_list):
                    f[i] <= s*x[i] + o
                    
    into glpk lp format    
    '''
    if A is not None:
        m,n = A.size
    else:
        m,n = 0,len(l)
    if C is not None:
        k,n = C.size
    else:
        k = 0

    lp = glpk.LPX()       
    lp.obj.maximize = True
    lp.cols.add(2*n)         # Append 2n columns (variables) [x,f]
    # box constraints
    for i,col in enumerate(lp.cols):      # Iterate over all columns
        if i < n:
            col.bounds = l[i],u[i]
        else:
            col.bounds = None, None
    # other constraints
    lp.rows.add(m+k)        
    for i,row in enumerate(lp.rows):  
        if i < m:
            row.bounds = None, b[i] 
        else:
            row.bounds = d[i-m], d[i-m]
    mat = []
    if A is not None:
        mat += numpy.concatenate((A,numpy.zeros((m,n))), axis=1).tolist() 
    if C is not None:
        mat += numpy.concatenate((C,numpy.zeros((k,n))), axis=1).tolist()
    mat = [x for row in mat for x in row]

    # objective
    lp.obj[:] = [0]*n + [1]*n # maximize sum of f
    # objective function constraints
    for offset in offsets:
        for o in offset:
            lp.rows.add(1)
            lp.rows[-1].bounds = None, o        
    for i,slope in enumerate(slopes):
        for j,s in enumerate(slope):
            xrow = [0]*n; frow = [0]*n
            xrow[i] = -s; frow[i] = 1
            mat = mat + xrow + frow
    lp.matrix = mat
    return lp             
              
def maximize_fapx_cvxopt( node, problem, scoop=False):
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
    
    scoop = True optionally dumps all problem parameters into a file
    which can be parsed and solved using the scoop second order cone 
    modeling language
    '''
    n = len(node.l);
    l = matrix(node.l); u = matrix(node.u)

    x = problem.variable
    constr = problem.constr
        
    # add box constraints
    box = [x[i]<=u[i] for i in xrange(n)] + [x[i]>=l[i] for i in xrange(n)]

    # find approximation to concave envelope of each function
    (fapx,slopes,offsets,fapxs) = utilities.get_fapx(node.tight,problem.fs,l,u,y=x)
    
    if problem.check_z:
        utilities.check_z(problem,node,fapx,x)
        
    if scoop:
        utilities.scoop(p,node,slopes,offsets)
    
    obj = sum( fapx )
    o = op(obj,constr + box)
    try:
        o.solve(solver = 'glpk')
    except:
        o.solve()
    if not o.status == 'optimal':
        if o.status == 'unknown':
            raise ImportError('Unable to solve subproblem. Please try again after installing cvxopt with glpk binding.')
        else:
            # This node is dead, since the problem is infeasible
            return False
    else:
        # find the difference between the fapx and f for each coordinate i
        fi = numpy.array([problem.fs[i][0](x.value[i]) for i in range(n)])
        fapxi = numpy.array([list(-fun.value())[0] for fun in fapx])
        #if verbose: print 'fi',fi,'fapxi',fapxi
        maxdiff_index = numpy.argmax( fapxi - fi )
        results = {'x': list(x.value), 'fapx': -list(obj.value())[0], 'f': float(sum(fi)), 'maxdiff_index': maxdiff_index}
        return results
        
def maximize_fapx_glpk( node, problem, verbose = False ):
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
    n = len(node.l);
    l = node.l; u = node.u
    
    # find approximation to concave envelope of each function
    (slopes,offsets,fapxs) = utilities.get_fapx(node.tight,problem.fs,l,u)
    
    # verify correctness of concave envelope
    if problem.check_z:
        utilities.check_z(problem,node,fapxs,x)
    
    # formulate concave problem as lp and solve    
    lp = sigopt2pyglpk(slopes = slopes,offsets=offsets,l=l,u=u,**problem.constr)
    lp.simplex()

    # find the difference between the fapx and f for each coordinate i
    xstar = [c.primal for c in lp.cols[:n]]
    fi = numpy.array([problem.fs[i][0](xstar[i]) for i in range(n)])
    fapxi = numpy.array([c.primal for c in lp.cols[n:]])
    maxdiff_index = numpy.argmax( fapxi - fi )
    results = {'x': xstar, 'fapx': lp.obj.value, 'f': float(sum(fi)), 'maxdiff_index': maxdiff_index}
    if verbose: print 'fi',fi,'fapxi',fapxi,results
    return results