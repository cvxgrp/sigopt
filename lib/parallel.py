'''
Utilities for parallelization of branch and bound algorithm for sigmoidal programming
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

from Queue import PriorityQueue
import time
import multiprocessing
import threading
import gc

class MaxQueue(object):
    ''' 
    A priority queue sorted in descending order instead of ascending order
    
    If maxlength > 0, queue keeps only the maxlength entries with highest values
    not memory efficient in python since small memory is not reused
    
        # reduce length to maxlength if queue is too long
        if self.maxlength > 0 and len(self) > self.maxlength + self.length_tol:
            with self.decrease_length:
                print 'Reducing length of MaxQueue by removing values less than',       
                print 'Used memory before = ', self.memory_mon.usage()
                new_pq = PriorityQueue()
                for i in range(self.maxlength - 1):
                    new_pq.put(self.pq.get())
                print -self.pq.get()[0], 'from consideration'
                self.pq = new_pq
                print 'Used memory after = ', self.memory_mon.usage()
    '''
    def __init__(self, maxlength = 0, length_tol = 1000):
        self.pq = PriorityQueue()
        self.decrease_length = multiprocessing.Lock()
        self.maxlength = maxlength
        self.length_tol = length_tol
        
    def put(self,tup):
        # add new item to queue            
        self.pq.put((-tup[0],tup[1]))

    def get(self,block=True):
        tup = self.pq.get(block)
        return (-tup[0],tup[1])
        
    def get_nowait(self):
        tup = self.pq.get_nowait()
        return (-tup[0],tup[1])        

    def __len__(self):
        return self.pq.qsize()
        
    def empty(self):
        return self.pq.empty()

class ThreadsafeBestNode(object):
    '''
    Threadsafe wrapper for best node
    '''
    def __init__(self,node):
        self.node = node
        self.lock = threading.Lock()
        
    def put(self,new_node):
        with self.lock:
            if new_node.LB > self.node.LB:
                self.node = new_node
                
    def get(self):
        return self.node
        
    def get_LB(self):
        return self.node.LB

class ProcessSafeInfo(object):
    '''
    Process safe shared information
    '''
    def __init__(self,best_node):
        manager = multiprocessing.Manager()
        self.best_node = manager.dict()
        self.best_node['best_node'] = best_node
        self.lock = multiprocessing.Lock()
        self.inbox = multiprocessing.Queue()
        self.outbox = multiprocessing.Queue()
        self.killed = multiprocessing.Value('i', 0)
        
    def put(self,new_node):
        with self.lock:
            if new_node.LB > self.best_node['best_node'].LB:
                self.best_node['best_node'] = new_node
                
    def get(self):
        return self.best_node['best_node']
        
    def get_LB(self):
        return self.best_node['best_node'].LB          
        
def threadsafe_function(fn):
    """decorator making sure that the decorated function is thread safe"""
    def new(*args, **kwargs):
        try:
            r = fn(*args, **kwargs)
            return r              
        except Exception as e:
            import sys
            info = sys.exc_info()
            raise info[1], None, info[2]
    return new

class ThreadedSolver(threading.Thread):    

    def __init__(self, problem):
        """
        Constructor.
        """
        threading.Thread.__init__(self)
        self.problem = problem
        self.verbose = True

    @threadsafe_function
    def run(self):
        while True:
            UB, node = self.problem.partition.get()
            if isinstance(node,str) and node == 'stop_computation':
                print 'stop computation recieved by',self.name
                break
            
            # record bounds
            self.problem._bounds.put((self.problem.threadsafe_best_node.get_LB(),UB))
            if self.verbose: 
                print 'LB',self.problem.threadsafe_best_node.get_LB(),'found by',self.name,'qsize',self.problem.partition.len()
                
            # stopping criterion
            if UB - self.problem.threadsafe_best_node.get_LB() < self.problem.tol: 
              # a tolerable solution has been found!
              # put current node back in the partition
              if self.verbose: print 'Stopping criterion reached by %s' % (self.name)
              self.problem.partition.put((UB,node))
              break
        
            # keep going
            for child in node.split(self.problem):
                if child.LB > self.problem.threadsafe_best_node.get_LB():
                    if self.verbose: 
                        print 'Node with best lower bound %.4f found by %s' \
                               % (child.LB, self.name)
                    self.problem.threadsafe_best_node.put(child)
                self.problem.partition.put((child.UB,child))

class ProcessedSolver(multiprocessing.Process):    

    def __init__(self, problem, shared, verbose = False, memory_limit = 0):
        """
        Constructor.
        """
        multiprocessing.Process.__init__(self)
        self.problem = problem
        self.shared = shared
        self.verbose = verbose
        self.memory_limit = memory_limit
        if memory_limit > 0:
            from memorymonitor import MemoryMonitor
            self.memmon = MemoryMonitor()    
        
    @threadsafe_function
    def run(self):
        if self.verbose:
            print self.name,'is working! at',time.time()
        while True:
            try:
                UB, node = self.shared.inbox.get(True,1)
            except:
                continue

            # stopping criterion
            if isinstance(node,str) and node == 'stop_computation':
                if self.verbose:
                    print 'stop computation recieved by',self.name,'at',time.time()
                break

            if self.verbose:
                print self.name,'found node with UB',UB,'at',time.time()
                
            # stopping criterion moved to master
            #if UB - self.shared.get_LB() < self.problem.tol: 
              # a tolerable solution has been found!
              # put current node back in the partition
              #if self.verbose: print 'Stopping criterion reached by %s' % (self.name)
              #self.shared.outbox.put((UB,node))
              #break
              
            # memory check
            if self.memory_limit > 0:
                usage = self.memmon.usage() / 1000 # in MB
                if usage > self.memory_limit:
                    # put node back so partition stays complete
                    self.shared.outbox.put((UB,node))
                    if self.verbose: 
                        print 'killing process',self.name,\
                            'because memory usage',usage,\
                            'exceeded',self.memory_limit
                    self.shared.killed.value = self.shared.killed.value + 1
                    break
        
            # keep going
            for child in node.split(self.problem):
                if child.LB > self.shared.get_LB():
                    if self.verbose: 
                        print 'Node with best lower bound %.4f found by %s' \
                               % (child.LB, self.name)
                    self.shared.put(child)
                self.shared.outbox.put((child.UB,child))

class Solver(multiprocessing.Process):
    """
    Solves branch and bound subproblems
    """

    def __init__(self, queue, bounds, LB, tol):
        """
        Constructor.
        """
        multiprocessing.Process.__init__(self)
        self.bounds = bounds
        self.LB = LB
        self.queue = queue
        self.tol = tol
        print self.name,id(self.queue)
        self.verbose = True
        
    @threadsafe_function
    def run(self):
        """Thread run method. 

        Solves bb subproblems and writes children back into queue after updating global bounds if necessary.
        """
        while True:
            neg_UB, node = self.queue.get()
            UB = -neg_UB
            if self.verbose: print 'Node with upper bound %f popped from list by %s' % (node.UB, self.name)

            # record bounds
            self.bounds.put((self.LB.value,UB))

            # stopping criterion
            if self.verbose: 
                print 'LB',self.LB.value,'found by',self.name
                print 'error',UB - self.LB.value,'found by',self.name
            if UB - self.LB.value < self.tol: 
              # a tolerable solution has been found!
              # put current node back on the queue so the queue always covers the domain
              if self.verbose: print 'Stopping criterion reached by %s' % (self.name)
              self.queue.put((-UB,node))
              break

            # keep going
            node.split()
            for child in node.children:
                if child.LB > self.LB.value:
                    if self.verbose: print 'Node with best lower bound %.4f found by %s' % (child.LB, self.name)
                    self.LB.value = child.LB
                self.queue.put((-child.UB,child))
            self.queue.task_done()
            print self.name,id(self.queue),self.queue.qsize()
        print self.name,id(self.queue),self.queue.qsize()

def run_parallel(problem,nthreads = 4):
    # enable locking on problem so that best node is never improperly overwritten
    problem.lock = multiprocessing.Lock()

    # sort nodes in descending order by upper bound
    problem.queue = PriorityQueue()
    problem.queue.put((-problem.top_node.UB,problem.top_node))
    threads = [Solver(problem) for i in range(nthreads)]
    try:
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    # exit gracefuly with best yet bounds if user is impatient
    except KeyboardInterrupt:
        for i in range(nthreads):
            queue.put((inf,'stop_computation'))
            
def run_parallel_processes():
    '''
    problem method that doesn't yet work.
    '''
    # enable locking on problem so that best node is never improperly overwritten
    bounds = multiprocessing.Queue()
    LB = multiprocessing.Value('d',-float("inf"))
    
    pq = self.queue #multiprocessing.sharedctypes.synchronized(self.queue)
    #pq = multiprocessing.Queue()
    #pq.put((-self.best_node.UB,self.best_node))
    threads = [parallel.Solver(pq,bounds,LB,self.tol) for i in range(self.nthreads)]
    try:
      for t in threads:
          print 'Starting thread',t.name
          t.start()
      # only need to join one thread since stopping condition is global,
      # but we'll sacrifice efficiency for tidyness
      for t in threads:
        t.join()
    
    # exit gracefuly with best yet bounds if user is impatient
    # XXX dosn't work consistently. consider using pool (ie pool.close(); pool.join()) instead
    except KeyboardInterrupt:
      print 'caught keyboard interrupt; stopping subprocesses'
      raise
      
    print [(id(t.queue),t.queue.qsize()) for t in threads]
    print id(self.queue),self.queue.qsize()
    print id(pq),pq.qsize()
    #self.queue = pq.get_obj()
    print self.queue.qsize()
    print LB.value
    #sys.exit()
    
    # find bounds    
    while not bounds.empty():
        next = bounds.get()
        self.history.bounds.append(next)
    # find best node
    while not self.queue.empty():
        print 'popping off',self.queue.qsize()
        UB,node = self.queue.get()
        print -UB,node
        if node.LB > self.best_node.LB:
            self.best_node = node
        o.solveself.queue.task_done()

if __name__ == '__main__':
    from knapsack_example_problems import bidding_example
    problem = bidding_example(n=36,type='winnings',tol=.001)
    problem.best_node = problem.top_node
    run_parallel(problem)
    print problem.history.bounds
    problem.best_node = lambda n=problem.best_node: n
    from figures import plot_best_node
    plot_best_node(problem)