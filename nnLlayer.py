import numpy as np
import math
import sys
#np.seterr(all='raise')

defaultLayers = ({'activation' : 'Tanh'  , 'size' : 12},
                 {'activation' : 'Tanh'  , 'size' :  6},
                 {'activation' : 'Tanh'   , 'size' :  4},
                 {'activation' : 'Sigmoid', 'size' :  2})

### class Layer
class Layer:
  def __init__(self, layerType, layerSize, prevLayerSize):
    self.type = layerType
    self.size = layerSize
    self.W = (np.random.randn(layerSize, prevLayerSize) * np.sqrt(1.0/prevLayerSize)).\
             astype(np.float64, copy=False)
    self.b = (np.zeros((layerSize, 1))).astype(np.float64, copy=False)
    self.adamVdW = np.zeros_like(self.W)
    self.adamVdb = np.zeros_like(self.b)
    self.adamSdW = np.zeros_like(self.W)
    self.adamSdb = np.zeros_like(self.b)
    self.adamT = 2
    pass

  def load(self, layerType, layerShape, W, b, VdW, Vdb, SdW, Sdb):
    self.type = layerType
    self.size = layerShape[0]
    self.W = np.reshape(W, (-1, layerShape[1])).astype(np.float64, copy=False)
    self.b = np.reshape(b, (-1, 1)).astype(np.float64, copy=False)
    self.adamVdW = np.reshape(VdW, (-1, layerShape[1])).astype(np.float64, copy=False)
    self.adamVdb = np.reshape(Vdb, (-1, 1)).astype(np.float64, copy=False)
    self.adamSdW = np.reshape(SdW, (-1, layerShape[1])).astype(np.float64, copy=False)
    self.adamSdb = np.reshape(Sdb, (-1, 1)).astype(np.float64, copy=False)
    pass

  def forward(self, Aprev):
    self.Aprev = Aprev
    self.Z = np.dot(self.W, Aprev) + self.b
    if   self.type == 'Sigmoid': self.A = self.sigmoid(self.Z)
    elif self.type == 'lReLU':   self.A = self.lrelu(self.Z)
    elif self.type == 'ReLU':    self.A = self.relu(self.Z)
    elif self.type == 'Tanh':    self.A = self.tanh(self.Z)
    else: assert(False),"Layer type activation is not implemented ({})".format(self.type)
    return self.A

  def backward(self, dA, batch_size, dZ=None):
    if   self.type == 'Sigmoid': g_primeZ = self.sigmoid(self.Z, True)
    elif self.type == 'lReLU':   g_primeZ = self.lrelu(self.Z,   True)
    elif self.type == 'ReLU':    g_primeZ = self.relu(self.Z,    True)
    elif self.type == 'Tanh':    g_primeZ = self.tanh(self.Z,    True)
    else: assert(False),"Layer type activation is not implemented ({})".format(self.type)
    if dZ is None: self.dZ = dA * g_primeZ
    else: self.dZ = dZ
    self.dAprev = np.dot(self.W.T, self.dZ)
    self.dW = np.dot(self.dZ, self.Aprev.T) / float(batch_size)
    self.db = np.sum(self.dZ, axis=1, keepdims=True) / float(batch_size)
    return self.dAprev, self.dW, self.db 

  def updateWeights(self, alfa):
    self.W -= alfa * self.dW
    self.b -= alfa * self.db
    pass

  def updateWeightsWithAdam(self, alfa, beta1=0.9, beta2=0.999, epsilon=1e-8):
    self.adamVdW = beta1 * self.adamVdW + (1. - beta1) * self.dW
    self.adamVdb = beta1 * self.adamVdb + (1. - beta1) * self.db
    self.adamSdW = beta2 * self.adamSdW + (1. - beta2) * np.power(self.dW, 2)
    self.adamSdb = beta2 * self.adamSdb + (1. - beta2) * np.power(self.db, 2)
    vCorrecteddW = self.adamVdW / (1. - np.power(beta1, self.adamT))
    vCorrecteddb = self.adamVdb / (1. - np.power(beta1, self.adamT))
    sCorrecteddW = self.adamSdW / (1. - np.power(beta2, self.adamT))
    sCorrecteddb = self.adamSdb / (1. - np.power(beta2, self.adamT))
    #self.adamT += 1
    self.W -= alfa * vCorrecteddW / (np.sqrt(sCorrecteddW) + epsilon)
    self.b -= alfa * vCorrecteddb / (np.sqrt(sCorrecteddb) + epsilon)
    pass

  def sigmoid(self,z,deriv=False):
    #try:
      if deriv:  #return np.exp(-z) / ((1.0 + np.exp(-z))**2)
        sig = self.sigmoid(z)
        return sig * (1. - sig)
      return 1.0 / (1.0 + np.exp(-z))
    #except:
    #  print np.amax(z)
    #  print np.exp(np.amax(z))
    #  print np.amin(z)
    #  print np.exp(np.amin(z))
    #  sys.exit()

  def lrelu(self,z,deriv=False):
    m = np.ones_like(z)
    m[np.where(z<=0)] = -0.01
    if deriv: return m
    return z * m

  def relu(self,z,deriv=False):
    m = np.ones_like(z)
    m[np.where(z<=0)] = 0.0
    if deriv: return m
    return z * m

  def tanh(self,z,deriv=False):
    if deriv: return 1. - np.tanh(z)**2
    return np.tanh(z)

### End of class Layer

### class L Layer NN
class LLayerNN:
  def __init__(self, featureVectorSize=1, layers=None):    # 'layers' specifies just hidden ones
    self.inputLayerSize = featureVectorSize
    if layers is None: self.initHiddenLayers(defaultLayers)
    else: self.initHiddenLayers(layers)
    self.outputLayer = Layer('Sigmoid', 1, self.hidden_layers[-1].size)
    self.norms = None
    pass

  def load(self, filename):
    with open(filename, 'r') as f:
      lines = f.read().splitlines()
      self.inputLayerSize = int(lines[0])
      if lines[1] != '':
        self.norms = []
        for pair in lines[1].split(';'):
          m, s = pair.split()
          self.norms.append((np.float64(m), np.float64(s)))
      self.hidden_layers = [ self.loadLayer(layer) for layer in lines[2:-1] ]
      self.outputLayer = self.loadLayer(lines[-1])
      return True
    return False

  def loadLayer(self, layer):
    layerType, layerShape, W, b, VdW, Vdb, SdW, Sdb = layer.split(';')
    layerShape = [int(el) for el in layerShape.split()]
    W   = np.asarray([np.float64(el) for el in   W.split()], dtype=np.float64)
    b   = np.asarray([np.float64(el) for el in   b.split()], dtype=np.float64)
    VdW = np.asarray([np.float64(el) for el in VdW.split()], dtype=np.float64)
    Vdb = np.asarray([np.float64(el) for el in Vdb.split()], dtype=np.float64)
    SdW = np.asarray([np.float64(el) for el in SdW.split()], dtype=np.float64)
    Sdb = np.asarray([np.float64(el) for el in Sdb.split()], dtype=np.float64)
    layer = Layer(layerType, layerShape[0], layerShape[1])
    layer.load(layerType, layerShape, W, b, VdW, Vdb, SdW, Sdb)
    return layer

  def save(self, filename):
    with open(filename, 'w') as f:
      f.write("{}\n".format(self.inputLayerSize))
      for n in self.norms[:-1]: f.write("{} {};".format(n[0], n[1]))
      f.write("{} {}\n".format(self.norms[-1][0], self.norms[-1][1]))
      for layer in self.hidden_layers: f.write(self.toString(layer) + '\n')
      f.write(self.toString(self.outputLayer))
      return True
    return False

  def toString(self, layer):
    s  = "{};{} {};".format(layer.type, layer.W.shape[0], layer.W.shape[1])
    for el in np.reshape(layer.W,       (1,-1))[0].tolist(): s += "{} ".format(el)
    s = s[:-1] + ';'
    for el in np.reshape(layer.b,       (1,-1))[0].tolist(): s += "{} ".format(el)
    s = s[:-1] + ';'
    for el in np.reshape(layer.adamVdW, (1,-1))[0].tolist(): s += "{} ".format(el)
    s = s[:-1] + ';'
    for el in np.reshape(layer.adamVdb, (1,-1))[0].tolist(): s += "{} ".format(el)
    s = s[:-1] + ';'
    for el in np.reshape(layer.adamSdW, (1,-1))[0].tolist(): s += "{} ".format(el)
    s = s[:-1] + ';'
    for el in np.reshape(layer.adamSdb, (1,-1))[0].tolist(): s += "{} ".format(el)
    s = s[:-1]
    return s

  def initHiddenLayers(self, layers):
    self.hidden_layers = []
    if len(layers) == 0: pass
    self.hidden_layers.append(Layer(layers[0]['activation'], layers[0]['size'], self.inputLayerSize))
    for i in range(1, len(layers)):
      self.hidden_layers.append(Layer(layers[i]['activation'], layers[i]['size'], layers[i-1]['size']))
    pass

  def cost(self, AL, Y):
    loss = (-Y) * np.log(AL) - (1.0 - Y) * np.log(1.0 - AL)
    return np.sum(loss) / Y.shape[1]

  def predict(self, X):
    A = X
    for layer in self.hidden_layers:
      A = layer.forward(A)
    A = self.outputLayer.forward(A)
    return A

  def train(self, X, Y, minibatchSize=128, alfa = 0.01, epochsNum = 100000):
    batchSize = X.shape[1]
    for j in xrange(epochsNum):
      permutation = list(np.random.permutation(batchSize))
      shuffled_X = X[:, permutation]
      shuffled_Y = Y[:, permutation].reshape((1,batchSize))
      minibatchesNum = int(math.floor(batchSize / minibatchSize))
      for k in range(0, minibatchesNum):
        minibatch_X = shuffled_X[:, k*minibatchSize:(k+1)*minibatchSize]
        minibatch_Y = shuffled_Y[:, k*minibatchSize:(k+1)*minibatchSize]
        costv = self.trainMiniBatch(minibatch_X, minibatch_Y, alfa)
        #if (j% 10000) == 0: print"\tError ({} from {}): {}".format(k, minibatchesNum, costv)
      if (j% 10000) == 0: print("Error after epoch {} (from {}): {}".format(j, epochsNum, costv))
    pass

  def trainMiniBatch(self, batchX, batchY, alfa):
    batchSize = batchX.shape[1]
    AL = self.predict(batchX)
    dA,_,_ = self.outputLayer.backward(dA=None, batch_size=batchSize, dZ=(AL-batchY))
    self.outputLayer.updateWeightsWithAdam(alfa)
    for layer in reversed(self.hidden_layers):
      dA,_,_ = layer.backward(dA=dA, batch_size=batchSize, dZ=None)
      layer.updateWeightsWithAdam(alfa)
    return self.cost(AL, batchY)

  def normalize(self, X):
    if self.norms is None:
      self.norms = []
      for j in range(X.shape[0]):
        mean = np.mean(X[j])
        X[j] -= mean
        std  = np.std(X[j])
        X[j] /= std
        self.norms.append((mean, std))
    else:
      assert(X.shape[0] == len(self.norms))
      for j in range(len(self.norms)):
        X[j] -= self.norms[j][0]
        X[j] /= self.norms[j][1]
    return X
  
### End of nnLlayer class

count = lambda ndarray: np.prod(ndarray.shape)

def evaluate(Yh, Y, threshold):
  assert(count(Y) > 0),"Provide evaluator with examples"
  assert(Yh.shape == Y.shape),"Shapes of resulting matrices must be the same"
  
  Yh[Yh >  threshold] = 1.0
  Yh[Yh <= threshold] = 0.0
  Yres = Yh * Y
  TP  = np.count_nonzero(Yres)
  FP  = np.count_nonzero(Yh) - TP
  pos = np.count_nonzero(Y)
  neg = count(Y) - pos
  FN  = pos - TP
  TN  = neg - FP
  if pos+neg > 0: acc = float(TP + TN)/float(pos + neg)
  else: acc = None
  if TP+FP > 0: pre = float(TP)/float(TP + FP)
  else: pre = None
  if TP+FN > 0: rec = float(TP)/float(TP + FN)
  else: rec = None
  f1s = 2.0*TP/float(2.0*TP + FN + FP) # 2.0*pre*rec/(pre+rec)
  if pos > 0: tpr = float(TP)/pos
  else: tpr = None
  if neg > 0: fpr = float(FP)/neg
  else: fpr = None
  acc = math.ceil(1000 * acc) / 10
  pre = math.ceil(1000 * pre) / 10
  rec = math.ceil(1000 * rec) / 10
  tpr = math.ceil(1000 * tpr) / 10
  fpr = math.ceil(1000 * fpr) / 10
  f1s = math.ceil(1000 * f1s) / 10
  return {'TP,FP,TN,FN':(TP,FP,TN,FN), 'pos,neg':(pos,neg), 'acc,pre,rec':(acc,pre,rec), 'tpr,fpr':(tpr,fpr), 'f1s':f1s}

if __name__ == '__main__':
  X = np.array([[0.5, 0.0, 0.5, 0.9, 0.2, 0.8, 0.0],
                [0.0, 1.0, 0.2, 0.9, 0.1, 0.7, 0.5],
                [0.4, 0.9, 0.4, 0.8, 0.1, 0.1, 0.0]])
  
  Y = np.array([[0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0]])  # 1 - for 1.0 < sum < 2.0

  shallowNN = LLayerNN(X.shape[0])
  X         = shallowNN.normalize(X)
  shallowNN.train(X, Y, minibatchSize=X.shape[1], epochsNum = 30000)

  Xtest = np.array([[0.3, 0.0, 0.5, 0.1, 0.8],
                    [0.9, 0.0, 0.2, 0.1, 0.7],
                    [0.4, 0.9, 0.4, 0.1, 0.6]])
  
  Ytest = np.array([[1.0, 0.0, 1.0, 0.0, 0.0]])

  Xtest = shallowNN.normalize(Xtest)

  print("-----------------")
  Yres = shallowNN.predict(X)
  costv = shallowNN.cost(Yres, Y)
  print("Train error: {}".format(costv))
  evalr = evaluate(Yres, Y, threshold=0.5)
  print("Train evaluation:")
  print(evalr)
  print(Yres)
  print(Y)

  shallowNN.save("test.nnweights")
  shallowNN2 = LLayerNN()
  shallowNN2.load("test.nnweights")
  print("-----------------")
  Yres = shallowNN2.predict(Xtest)
  costv = shallowNN2.cost(Yres, Ytest)
  print("Test error: {}".format(costv))
  evalr = evaluate(Yres, Ytest, threshold=0.5)
  print("Test evaluation:")
  print(evalr)
  print(Yres)
  print(Ytest)
