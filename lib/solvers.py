'''
LP solvers for sigmoidal programming problems.

Both cvxopt and cvxpy interfaces are included.
cvxopt sometimes refuses to solve an LP for technical reasons.
generically interior point methods such as cxvopt and cvxpy
produce lower LB than simplex solvers,
unless a random LP is solved as a refinement step (set rand_refine >= 1)
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

# XXX should work with cvxpy alone, or with glpk alone and rand_refine = 0
import numpy
import cvxpy
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
              
def maximize_fapx_cvxopt(node, problem, scoop=False):
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
    
    XXX obsolete usage of get_fapx ???
    '''
    n = len(node.l);
    l = matrix(node.l); u = matrix(node.u)

    x = problem.cvxopt_variable
    constr = problem.cvxopt_constr
        
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
        
def maximize_fapx_cvxpy(node, problem):
    '''
    Solves 
        maximize       sum(fapx(x)) 
        subject to     Ax <= b
                       Cx == d
                       node.l <= x <= node.u
    using cvxpy as the solver.
    
    fapx is calculated by approximating the functions fi given in fs
    as a piecewise linear upper bound on fi that is tight, 
    for each coordinate i, at the points in node.tight[i].
    fs should be a list of tuples containing the function and its derivative
    corresponding to each coordinate of the vector x.
  
    Returns a dict containing the optimal variable y as a list, 
    the optimal value of the approximate objective,
    and the value of sum(f(x)) at x.
    '''
    n = len(node.l);
    l = node.l; u = node.u

    (slopes,offsets,fapxs) = utilities.get_fapx(node.tight,problem.fs,l,u)
    
    A = problem.constr['A']
    b = problem.constr['b']
    C = problem.constr['C']
    d = problem.constr['d']

    ## Create cvxpy problem to solve
    # We write maximize sum(fapx(x)) as maximize sum(f),
    # where f[i] <= pwl upper bound on fi
    x = cvxpy.Variable(n)
    f = cvxpy.Variable(n)
    objective = cvxpy.Maximize(sum(f))
    constraints = [x <= problem.u, x >= problem.l]
    if A is not None and b is not None:
        constraints.append(A*x <= b)
    if C is not None and d is not None:
        constraints.append(C*x == d)
    for i,(slope_list,offset_list) in enumerate(zip(slopes,offsets)):
        for s,o in zip(slope_list,offset_list):
            constraints.append(f[i] <= s*x[i] + o)
        
    lp = cvxpy.Problem(objective,constraints)
    phat = lp.solve()
    # check if problem is infeasible
    if not lp.status == 'optimal':
        return False
    
    if problem.check_z:
        utilities.check_z(problem,node,fapx,x)
    
    # find the difference between the fapx and f for each coordinate i
    fi_of_x = numpy.matrix(map(lambda (fi,xi): fi[0](xi), zip(problem.fs,x.value)))
    maxdiff_index = numpy.argmax( f.value - fi_of_x)
    print "maxdiff_index = ", maxdiff_index
    results = {'x': list(x.value), 'fapx': phat, 'f': sum(fi_of_x), 'maxdiff_index': maxdiff_index}
    
    if problem.rand_refine > 0:
        rand_results = random_lp_cvxpy(problem,node,x.value,
                                       phat,numpy.sum(fi_of_x),slopes,offsets)
        if rand_results:
            results = rand_results
    
    return results
        
def random_lp_cvxpy(problem,node,xhat,phat,f_of_xhat,slopes, offsets):
    '''
    returns None if no solution of any random LP is better than original xhat
    '''
    n = len(problem.l)
    A = problem.constr['A']
    b = problem.constr['b']
    C = problem.constr['C']
    d = problem.constr['d']

    x = cvxpy.Variable(n)
    f = cvxpy.Variable(n)
    random_solutions = []; improved = False
    for i in range(problem.rand_refine):
        w = numpy.matrix(numpy.random.randn(n))
        objective = cvxpy.Minimize(w*x)
        constraints = [x <= problem.u, x >= problem.l]
        if A is not None and b is not None:
            constraints.append(A*x <= b)
        if C is not None and d is not None:
            constraints.append(C*x == d)
        for i,(slope_list,offset_list) in enumerate(zip(slopes,offsets)):
            for s,o in zip(slope_list,offset_list):
                # Ax <= b means f[i] - s*x[i] <= o so f[i] <= s*x[i] + o
                constraints.append(f[i] <= s*x[i] + o)
        constraints.append(sum(f) == phat)
            
        p = cvxpy.Problem(objective,constraints)
        p.solve()
        xtilde = x.value
        
        fi_of_xtilde = numpy.matrix(map(lambda (fi,xi): fi[0](xi), zip(problem.fs,xtilde))).T
        if numpy.sum(fi_of_xtilde) > f_of_xhat:
            improved = True
            xhat = xtilde
            fi_of_xhat = fi_of_xtilde
            f_of_xhat = sum(fi_of_xtilde)
            fhati_of_xhat = f.value
            
    if improved:
        maxdiff_index = numpy.argmax( fhati_of_xhat - fi_of_xhat)
        return {'x': list(xhat), 'fapx': phat, 'f': f_of_xhat, 'maxdiff_index': maxdiff_index}
    else:
        return None