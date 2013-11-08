'''
example sigmoidal programming problems

bidding: optimal bidding
ilp: integer linear programming
num: network utility maximization
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

from __future__ import division
from bb import Problem
import functions
import cvxopt, numpy

def ilp(n=20,m=5,seed=3,tol=.01):
  '''
  Constructs an integer linear programming problem
  as a sigmoidal program
  
  maximize sum_i (|x_i-.5| - .5)
  st       Cx=d, 
           0<=x<=1   
  
  Solution to this SPP has value 0 iff there is a vector x \in {0,1}^n
  such that Ax = b
  
  A \in \reals^{m \times n}
  '''
  # generate x\in \{0,1\}^n st A*x = b
  numpy.random.seed(seed)
  C = numpy.random.normal(0,1,(m,n))
  x0 = numpy.ceil(numpy.random.uniform(0,2,n))-1
  d = (numpy.dot(C,x0))

  fs = [(functions.be_01,functions.be_01p,1)]*n
  
  # Box constraints on x
  l = [0]*n
  u = [1]*n
  name = 'ilp_m=%d_n=%d'%(m,n)
  return Problem(l,u,fs,tol=tol,name=name,C=C,d=d)

def bidding(n=36,B=.5,tol=.01,type='profit'):
  '''
  Constructs an optimal bidding problem
  as a sigmoidal program
  
  type='profit':
      maximize sum_i (v_i - x_i)*(logistic(a_i*x_i+b_i)-logistic(b_i))
      st       sum(x) <= B*sum(v), 
               0<=x<=v
  type='winnings':
      maximize sum_i v_i(logistic(a_i*x_i+b_i)-logistic(b_i))
      st       sum(x) <= B*sum(v), 
               0<=x<=B*n
  type='pretty':
      maximize sum_i v_i logistic(x_i)
      st       sum(x) <= .3n, 
               -.3n<=x<=.3n
  '''
  A = numpy.ones((1,n))
  v = 4*numpy.random.rand(n)
  v = sorted(v.tolist())
  if type =='profit':
      c =  [B*sum(v)]
      aa = [10]*n
      bb = [-3*vi for vi in v]
      # convex-concave transition occurs where a(v-x)(1-2logistic(ax+b))=2,
      # ie near -b/a-2, so long as -b/a<v
      fs = [(lambda x,i=i,a=a,b=b: (v[i]-x)*(
                        functions.logistic(a*x+b)-functions.logistic(b)), 
             lambda x,i=i,a=a,b=b: a*(v[i]-x)*functions.logistic_prime(a*x+b) - 
                        (functions.logistic(a*x+b)-functions.logistic(b))) \
            for (i,a,b,vi) in zip(range(n),aa,bb,v)]
      l = [0]*n
      u = v
  elif type == 'winnings':
      int = 2+3*numpy.random.rand(n)
      int = sorted(int.tolist())
      c = cvxopt.matrix([B*n],tc='d')
      fs = [(lambda x,vi=vi,inti=inti: vi * functions.logistic(x-inti),
             lambda x,vi=vi,inti=inti: vi * functions.logistic_prime(x-inti)) \
            for vi,inti in zip(v,int)]
      l = [0]*n
      u = [B*n]*n
  elif type == 'pretty':
      c = [.3*n]
      fs = [(lambda x,vi=vi: vi * functions.logistic(x),
             lambda x,vi=vi: vi * functions.logistic_prime(x),
             0) \
            for vi in v]
      l = [-.3*n]*n
      u = [.3*n]*n
  elif type == 'admit':
      c = cvxopt.matrix([B*n],tc='d')
      fs = [(functions.admit,functions.admit_prime,1)]*n
      l = [0]*n
      u = [2]*n
  name = 'bidding_%s_B=%.1f_n=%d'%(type,B,n)
  return Problem(l,u,fs,A=A,b=c,tol=tol,name=name)  

def num(n=64,m=64,s=5,seed=3,opt='random',func='admit',tol=.01):
  '''
  Constructs a network utility maximization problem
  as a sigmoidal program
  
  maximize sum_i f_i(x_i)
  st       Ax <= c, 
           0<=x
           
  matrix A \in \reals^{m \times n} has s entries per row, on average
  capacities c are s/2 for every edge
  
  opt controls how the matrix A is chosen
      'ring': ring topology (m=n)
      'local': (m=n) prob 1/2 that flow i uses edge j for j \in [i,i+2s], 0 else
      'random': prob s/m that flow i uses edge j
  
  func controls how the function f is chosen
      'admit': admittance function (approximation to step function 
               with step of size 1 at x=1)
      'quadratic': f(x) = x^2
  '''
  cvxopt.setseed(seed)
  
  ## Set graph topology
  if opt == 'ring':
    m=n
    A = cvxopt.matrix(([1]*s+[0]*(n-s+1))*(n-1)+[1],(n,n),tc='d')
    c = cvxopt.matrix([s/2]*n,tc='d')
  elif opt == 'local':
    m=n
    probs = ([1]*s+[0]*(n-s+1))*(n-1)+[1]
    A = cvxopt.matrix([round(x*numpy.random.rand()) for x in probs],(n,n),tc='d')
    c = cvxopt.matrix([s/2]*n,tc='d')
    # print 'There are',sum(A),'edges used in this NUM problem'
  elif opt == 'random':
    A = cvxopt.matrix([round(.5/(1-float(s)/m)*x) for x in numpy.random.rand(m*n)]
,(m,n),tc='d')
    #print 'There are',sum(A),'edges used in this NUM problem'
    c = cvxopt.matrix([s/2]*m,tc='d')
  else:
    raise ValueError('unknown option %s'%opt)
  
  ## Set utility function
  # admission
  if func=='admit':
    fs = [(functions.admit, functions.admit_prime, 1) for i in range(n)]
  # quadratic
  elif func=='quadratic':
    fs = [(lambda x: pow(x,2), lambda x: 2*x, 0) for i in range(n)]
  else:
    raise ValueError('unknown function %s'%func)
  l = [0]*n
  u = [s/2]*n
  tol = tol
  name = 'num_%s_%s_n=%d_m=%d_s=%d'%(opt,func,n,m,s)
  return Problem(l,u,fs,A=A,b=c,tol=tol,name=name)    
         
if __name__ == '__main__':
    import unit_tests
    unit_tests.main()