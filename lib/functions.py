import math

def logistic(z):
  	return 1/(1+math.exp(-z))

def logistic_prime(z):
  return logistic(z)*(1-logistic(z))
  
# define a sigmoidal function that is <=0, and is 0 iff x \in {0,1}
def be_01(x):
    if x <= .5:
        return -x
    else:
        return x-1
def be_01p(x):
    if x <= .5:
        return -1
    else:
        return 1
  
def line(x):
    return x*2
    
def convex(x):
    return math.pow(x,2)
    
def concave(x):
    return -math.pow(x,2)
    
def convexconcave1(x):
    if x>0:
        return -math.pow(x-5,2)+25
    else:
        return math.pow(x+5,2)-25

def admit(x,threshold=1,eps=.1):
    if x>threshold:
        return 1
    elif x<threshold-eps:
        return 0
    else:
        return 1-(threshold-x)/eps
        
def admit_prime(x,threshold=1,eps=.1):
    if x>threshold:
        return 0
    elif x<threshold-eps:
        return 0
    else:
        return 1/eps
        
def phi(x):
    return .5*(1 + erf(x/math.sqrt(2)))

def normal(x):
    return 1/(math.sqrt(2)*math.pi)*math.exp(-pow(x,2)/2.0)
        
def convexconcave2(x):
    if x>0:
        return math.sqrt(.5*x)
    else:
        return -math.sqrt(-.5*x)
        
def convexconcave3(x):
    if x>0:
        return math.sqrt(.5*x)
    else:
        return .2*x
        
def convexconcave4(x):
    if x>0:
        return -.5*math.pow(x-2,2)+2
    else:
        return -math.sqrt(-x)
        
def economies(x):
    if x<15:
        return -math.pow(x,3) + 6*math.pow(x,2) - 9*x - 1
    else:
        return -x + 7    
