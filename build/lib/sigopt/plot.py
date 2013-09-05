'''
Plotting library for sigmoidal programming problems.

plot_convergence: plot convergence of sigmoidal program

plot_node: plot concave envelope over node, along with solution to concave problem on that node

plot_best_node: plot concave envelope over best node found so far, along with solution to concave problem on that node
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
import numpy
import math
import matplotlib.pyplot as plt
import cvxopt
import utilities

def plot_convergence(p,fn='',verbose=True):
    '''Plots the upper and lower bounds on the solution
    of the sigmoial programming problem p'''    
    fig = plt.figure()
    
    ax = fig.add_subplot(1,1,1)
    UB = [t[1] for t in p.bounds]
    ax.plot(UB)
    LB = [t[0] for t in p.bounds]
    ax.plot(LB)
    ax.xaxis.set_label_text('Iteration')
    ax.yaxis.set_label_text('Bound')
    
    if not fn:
        fn = '%s_convergence.eps'%p.name
    if verbose: 
        print 'Saving plot of history in',fn
    fig.savefig(fn)
    return fig
            
def plot_node(p,n,fn='',ncol=None,annotation=None,coords_to_plot=None,plot_opt = 1,verbose=True):
    '''
    Plots node n and its concave envelope for sigmoidal programming problem p
    
    Each graph represents one variable and its associated objective function (solid line) 
    and concave approximation (dashed line) on the rectangle that the node represents.
    The solution $x^\star_i$ for each variable is given by 
    the x coordinate of the red X.
    The node's rectangle is bounded by the solid grey lines
    and the endpoints of the interval.
    The red line gives the error between the value of the concave approximation $\hat{f}$
    and the sigmoidal function $f$ at the solution $x\star$.
    '''
    if annotation is None: annotation = {}
    if n is None:
        raise ValueError('Node is empty --- cannot plot. \
                          Perhaps the problem is infeasible?')
                          
    if hasattr(n,'x') and n.x:
        plot_opt = plot_opt
    else:
        if plot_opt:
            plot_opt = 0
            if verbose: print 'No solution found for node. Not plotting optimal point.'
            
    tight = [utilities.find_concave_hull(li,ui,fi,[],p.sub_tol) for li,ui,fi in zip(n.l,n.u,p.fs)]
    fapx = utilities.get_fapx(tight, p.fs, n.l, n.u, tol = p.sub_tol)[-1]
    
    if not ncol:
        ncol = math.floor(math.sqrt(len(p.l)))
        nrow = math.ceil(len(p.l)/float(ncol))
    fig = plt.figure()
    if 'subtitle' in annotation.keys():
    	fig.subplots_adjust(hspace = .8)
    if coords_to_plot is None:
        coords_to_plot = range(len(p.l))
    for i in coords_to_plot:
        ax = fig.add_subplot(nrow,ncol,i+1)
        if 'subtitle' in annotation:
            if annotation['subtitle'] == 'numbered':    
                ax.title('%d'%i)
            else:
                ax.set_title(annotation['subtitle'][i])
                
        # plot f
        x = list(numpy.linspace(p.l[i],p.u[i],300))
        ax.plot(x,[p.fs[i][0](xi) for xi in x],'b-')

        # plot fapx, only over the node n
        nx = list(numpy.linspace(n.l[i],n.u[i],300))
        fapxi=[]
        for xi in nx:
            fapxi.append(fapx[i](xi))
        ax.plot(nx,fapxi,'c--')

        # plot node boundary
        if n.l[i]>p.l[i]: 
            ax.axvline(x=n.l[i], linewidth=.5, color = 'grey')
        if n.u[i]<p.u[i]: 
            ax.axvline(x=n.u[i], linewidth=.5, color = 'grey')

        # plot optimal point
        if plot_opt:
            ax.plot([n.x[i]],[p.fs[i][0](n.x[i])],'rx')            
            ax.plot([n.x[i]],[fapx[i](n.x[i])],'rx')   
            # plot error
            ax.vlines(n.x[i],fapx[i](n.x[i]),p.fs[i][0](n.x[i]),'r')
        
        ax.margins(.05)
        ax.axis('off')
        #ax.axis('equal')

    if not fn:
        fn = p.name+'_best_node.eps'
    if verbose:
        print 'Saving figure in',fn
    fig.savefig(fn)
    return fig
    
def plot_best_node(p,**kwargs):
    return plot_node(p,p.best_node,**kwargs)

if __name__ == '__main__':
    import examples
    p = examples.ilp(n=25,m=5)
    p.solve(maxiters=10) 
    plot_best_node(p,annotation={},fn='ilp.png')