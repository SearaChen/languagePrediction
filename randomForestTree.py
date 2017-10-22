from __future__ import division
import random
import operator
import numpy
import sys

def checkEqual(l):
    return all(l[0] == rest for rest in l)

def entropy(y):
    y = numpy.asarray(y,dtype='uint8')
    counts = numpy.bincount(y.ravel())
    counts = counts[counts > 0]
    probs = counts / y.size
    return -numpy.sum(probs * numpy.log2(probs))
    
def findBestTest(f,y):
    n=len(y)
    if checkEqual(f):
        return f[0] , entropy(y)
    f,y = zip(*sorted(zip(f,y)))
    index, value = min([(i,(i*entropy(y[:i-1]) + (n-i)*entropy(y[i:]))/n) 
                                for i in range(0,n) if f[i] != f[i-1]], key=operator.itemgetter(1))
    return (f[index]+f[index-1])/2 , value

class node:
    
    random_m = None
    
    def __init__(self,M, decTree):
        if(len(M) != 2):
            print("M=",M)
            sys.exit('Faulty')
        self.decTree = decTree
        self.M=M
        self.label = 1
        self.feature = None
        self.test = None
        self.entropy = 0
        self.leftBranch = None
        self.rightBranch = None
        
    def getLeftBranch(self):
        return self.leftBranch
    
    def getRightBranch(self):
        return self.rightBranch

    def setLeftBranch(self, leftNode):
        self.decTree.tree.append(leftNode)
        self.leftBranch = leftNode
    
    def setRightBranch(self, rightNode):
        self.decTree.tree.append(rightNode)
        self.rightBranch = rightNode

    def buildNode(self):
        x=list(self.M[0])
        y=list(self.M[1])
        print "new node"
        if checkEqual(y): #at leaf
#            print("true if")
            self.label = y[0]
            self.feature = None
            self.test = None
            self.entropy = 0
            self.leftBranch = None
            self.rightBranch = None
        else:
#            print("else")
            self.feature, self.test, self.entropy = min(
                    [(i,)+findBestTest(f,y) for i,f in
                     random.sample(list(enumerate(map(list, zip(*x)))),node.random_m)], key=operator.itemgetter(2))
#                    list(enumerate(map(list, zip(*x))))], key=operator.itemgetter(2))
        
        if(entropy(y) == self.entropy):
            self.label = max(y,key=y.count) 
            self.feature = None
            self.test = None
            self.leftBranch = None
            self.rightBranch = None
        else:
            n = len(y)
            if(len(zip(*[(x[i][self.feature],y[i]) for i in range(n) if x[i][self.feature] >= self.test])) == 0 or len(zip(*[(x[i][self.feature],y[i]) for i in range(n) if x[i][self.feature] < self.test])) == 0):
                print(self.feature, self.test, self.entropy, ">",
                      y,"Right:",[row[self.feature] for row in x if row[self.feature]>=self],"Left:"
                                      ,[row[self.feature] for row in x if row[self.feature]<self])
            self.setRightBranch(node(zip(*[(x[i],y[i]) for i in range(n) if x[i][self.feature] >= self.test]),self.decTree))
            self.setLeftBranch(node(zip(*[(x[i],y[i]) for i in range(n) if x[i][self.feature] < self.test]),self.decTree))

class decTree:
    
    
    def __init__(self,M):
        self.tree=list()
        self.pointerSoFar=0
        self.tree.append(node(M,self))
        while(self.pointerSoFar != len(self.tree)):
            self.tree[self.pointerSoFar].buildNode()
            self.pointerSoFar += 1 
                            
    def evaluateExample(self,x):
        node_=self.tree[0]
        while(node_.feature is not None): #at leaf
            #if(node_.feature is None or node_.test is None):
            if x[node_.feature] >= node_.test:
                node_ = node_.getRightBranch()
            else:
                node_ = node_.getLeftBranch()
        return node_.label
        
#     def getData(self):
#         return self.data
    
#     def setData(self,newdata):
#         self.data = newdata
        
        
class randomForestTree:
    def __init__(self,x,y,m,K):
        node.random_m = m
        n = len(y)
        secure_random = random.SystemRandom()
        self.trees = [decTree(zip(* [(x[i],y[i]) for i in [secure_random.choice(range(n)) for r in range(n)]] )) for i in range(K)]
        #self.trees = [decTree(zip(*[(x[i],y[i]) for i in range(n)])) for i in range(K)]
        
    def predict(self,x):
        votes = [t.evaluateExample(x) for t in self.trees]
        return int(max(votes, key=votes.count)[0])

    def score(self,x,y):
        n=len(y)
        return len([True for i in range(n) if self.predict(x[i]) == y[i][0]]) / n

X = [[0, 0], [1, 1]]
Y = [[0], [1]]
z = [[-1,-1]]
model1 = randomForestTree(X,Y,2,20)
print(model1.score(z,[[1]]))
