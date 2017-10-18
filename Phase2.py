
# coding: utf-8

# In[120]:


import random
import operator
import numpy

def checkEqual(l):
    return all(l[0] == rest for rest in l)

def entropy(y):
    y = numpy.asarray(y,dtype='uint8')
    counts = numpy.bincount(y)
    counts = counts[counts > 0]
    probs = counts / y.size
    return -numpy.sum(probs * numpy.log2(probs))
    

def findBestTest(f,y):
    f,y = zip(*sorted(zip(f,y)))
    n=len(y)
    index, value = min([(i,(i*entropy(y[:i-1]) + (n-i)*entropy(y[i:]))/n) 
                                for i in range(1,n) if y[i]!=y[i-1]], key=operator.itemgetter(1))
    return (f[index]+f[index-1])/2 , value

class node:
    random_m = None
    def __init__(self,x,y):
        if checkEqual(y): #at leaf
            self.label = y[0]
            self.feature = None
            self.test = None
            self.entropy = 0
            self.leftBranch = None
            self.rightBranch = None
        else:
            self.feature, self.test, self.entropy = min(
                    [(i,*findBestTest(f,y)) for i,f in enumerate(
                            #random.sample(list(map(list, zip(*x))),self.random_m))], key=operator.itemgetter(2))
                            list(map(list, zip(*x))))], key=operator.itemgetter(2))
            
            n = len(y)
            self.setRightBranch(node(*zip(*[(x[i],y[i]) for i in range(n) if x[i][self.feature] >= self.test])))
            self.setLeftBranch(node(*zip(*[(x[i],y[i]) for i in range(n) if x[i][self.feature] < self.test])))

            
    def evaluateExample(self,x):
        if self.feature is None: #at leaf
            return self.label
        elif x[self.feature] >= self.test:
            return self.getRightBranch().evaluateExample(x)
        else:
            return self.getLeftBranch().evaluateExample(x)
            
#     def getData(self):
#         return self.data
    
#     def setData(self,newdata):
#         self.data = newdata
        
    def getLeftBranch(self):
        return self.leftBranch
    
    def getRightBranch(self):
        return self.rightBranch

    def setLeftBranch(self, leftBranch):
        self.leftBranch = leftBranch
    
    def setRightBranch(self, rightBranch):
        self.rightBranch = rightBranch
        
class randomForestTree:
    def __init__(self,x,y,m,K):
        node.random_m = m
        n = len(y)
        #self.trees = [node(*zip(*[(x[i],y[i]) for i in random.choices(range(n),k=n)])) for i in range(K)]
        self.trees = [node(*zip(*[(x[i],y[i]) for i in range(n)])) for i in range(K)]
        
    def predict(self,x):
        votes = [t.evaluateExample(x) for t in self.trees]
        return max(set(votes), key=votes.count)


# In[141]:


#import randomForestTree

X = [[0, 0], [1, 1]]
Y = [0, 1]
z = [0,0]
model = randomForestTree(X,Y,2,1)
model.predict(z)

