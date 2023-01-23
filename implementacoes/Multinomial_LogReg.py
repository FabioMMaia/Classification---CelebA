# -*- coding: utf-8 -*-
""" Origem: Regressão Logística Multinomial.ipynb

**Objetivo:**
Implementar o ´código regressão logística com taxa de aprendizado variável (usar método de bisseção).
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


"""# 01. Implementação da Regressão Logística Multinomial - com taxa de aprendizado constante

"""

class MultinomialLogReg():

  def __init__(self,alpha=.005, max_iter=1e8):
    self.alpha=alpha
    self.max_iter=max_iter
    self.theta = 1e-3

  def fit(self, X,y):
    self.X=np.array(X)
    self.y=np.array(y)
    self.m, self.p=X.shape
    self.classes = (np.unique(y))
    self.K = len(self.classes)
    self.adjust_matrices()
    self.memory={}

    # Inicializa w randomicamente
    
    w = np.random.random((self.K,self.p+1))
    it=0
    norm_grad=1e4

    while it<self.max_iter and norm_grad>self.theta: 

      # Calcula epsilon produto de w por X
      eps = np.dot(self.X_, w.T)

      # Aplica softmax
      yhat = self.soft_max(eps)
      
      # Calcula o custo (cross-entropia)
      J= -(self.y_*np.log(yhat)).sum()

      # Calcula o gradiente e respectivamente sua norma
      gradE = np.dot((self.y_ - yhat).T, self.X_)
      norm_grad = self.norm_matrix(gradE)
      
      # Atualiza w (na direção ascendente - maxima verossimilhança -- para tornar um problema de minimização deve se tomar o sinal contrário de gradE)
      w = w + self.alpha*gradE
      
      # Retém na memoria
      self.memory[it] = (J, gradE)
      it+=1

      if it%1e4==0:
        print('it:{}, norm_grad:{}, J:{}'.format(it, norm_grad, J))

    print('it:{}, norm_grad:{}, J:{}'.format(it, norm_grad, J))
    self.w = w

  def predict(self, Xpred):
    Xpred_ = np.c_[Xpred, np.ones(Xpred.shape[0])]
    eps = np.dot(Xpred_, self.w.T)
    ypred = self.soft_max(eps)
    return ypred

  def predict_proba(self, Xpred):
    Xpred_ = np.c_[Xpred, np.ones(Xpred.shape[0])]
    eps = np.dot(Xpred_, self.w.T)
    ypred = self.soft_max(eps)
    return ypred

  def predict(self, Xpred):
    return self.predict_proba(Xpred).argmax(axis=1)

  def adjust_matrices(self):
    # adiciona 1 à esqueda de X
    self.X_ = np.c_[ self.X, np.ones(self.m)]

    # faz one hot enconding de y
    y_ = np.zeros((self.m, self.K))
    for i in range(0, self.m):
      for k in range(0,self.K):
        y_[i, k] = self.y[i]==self.classes[k]

    # Converte de bool para int
    self.y_ = y_.astype(int)

  def soft_max(self, M):
    return np.e**(M)/(np.e**(M)).sum(axis=1).reshape(-1,1)

  def norm_matrix(self,M):
    return np.linalg.norm(M.reshape(-1,1))


class MultinomialLogReg_bissecao(MultinomialLogReg):

  def __init__(self, max_iter=1e8, theta=1e-4 , itmax_bissec=20,  verbose=False):
    self.verbose=verbose
    self.max_iter=max_iter
    self.theta = theta
    self.itmax_bissec=itmax_bissec

  def calculate_grad(self, w):
      
      # Calcula epsilon produto de w por X
      eps = np.dot(self.X_, w.T)

      # Aplica softmax
      yhat = self.soft_max(eps)
      
      # Calcula o custo (cross-entropia)
      J= -(self.y_*np.log(yhat)).sum()

      # Calcula o gradiente e respectivamente sua norma
      gradE = np.dot((self.y_ - yhat).T, self.X_)

      # Retorno o -gradE pois queremos minimizar o custo - ao se maximizar E estamos minimizando -E - o que é equivalente a minimizar a entropia cruzada J
      return (-gradE, yhat, J)

  def calc_alfa_bissecao(self, x,d):

    itmax= self.itmax_bissec

    # Entrada: vetor x e direção d (gradiente)
    # Retorna: melhor valor de alpha
    # Recebe o vetor x do espaço que devemos encontrar o valor de mínimo na função (melhor alpha)

    verbose = self.verbose
    alfa = np.random.random(1)[0]
    xn = x + alfa*d
    g, _, _ = self.calculate_grad(xn)
    hl = np.sum(g*d)

    if verbose:
      print('primeiro hl:', hl)
    alfa_l = 0

    while hl<0:
      if verbose:
        print('hl:',hl)
  
      alfa_l = alfa
      alfa=alfa*2
      xn = x + alfa*d
      g, _, _ = self.calculate_grad(xn)
      hl = np.sum(g*d)

    alfa_u = alfa
    alfa_m = (alfa_l + alfa_u)/2
    it=0

    if verbose:
      print('it:{} hl:{} x:{} xn:{} g:{} alfa_l:{} alfa_m:{} alfa_u:{}'.format(it,hl,x,  xn , g,   alfa_l, alfa_m, alfa_u))

    while it<itmax:
      it+=1
      xn=x+alfa_m*d
      g, _, _ ==self.calculate_grad(xn)
      hl=np.sum(g*d)

      if hl<self.theta:
        return alfa_m, hl

      elif hl>0:
          alfa_u = alfa_m

      else:
        alfa_l = alfa_m

      if verbose:
        print('it:{} hl:{} x:{} xn:{} g:{} alfa_l:{} alfa_m:{} alfa_u:{}'.format(it,hl,x,  xn , g,   alfa_l, alfa_m, alfa_u))

      alfa_m = (alfa_l + alfa_u)/2 

    return alfa_m, hl

  

  def fit(self, X,y):
    self.X=np.array(X)
    self.y=np.array(y)
    self.m, self.p=X.shape
    self.classes = (np.unique(y))
    self.K = len(self.classes)
    self.adjust_matrices()
    self.memory={}

    # Inicializa w randomicamente
    w = np.random.random((self.K,self.p+1))

    # Calcula o gradiente
    gradE, yhat, J = self.calculate_grad(w)
    d= -gradE
    norm_grad = self.norm_matrix(gradE)
    it=0
    print('iteracao:',it, 'norm_grad:', norm_grad )

    while it<self.max_iter and norm_grad>self.theta: 

      it+=1

      # Calcula o valor de alpha
      alpha_b, hl = self.calc_alfa_bissecao(w, d)

      # Atualiza w
      w = w + alpha_b*d

      # atualiza o valor do gradiente
      gradE, yhat, J = self.calculate_grad(w)

      # atualiza o valor de d
      d = - gradE

      # Calcula a norma do vetor gradiente
      norm_grad = self.norm_matrix(gradE)

      # Retém na memoria
      self.memory[it] = (J, gradE,alpha_b, hl)
      
      if it%1e3==0:
        print('it:{}, norm_grad:{}, J:{}, alpha_b:{}, hl:{}'.format(it, norm_grad, J,alpha_b, hl))

    print('it:{}, norm_grad:{}, J:{}, alpha_b:{}, hl:{}'.format(it, norm_grad, J,alpha_b, hl))
    self.w = w


  def adjust_matrices(self):
    # adiciona 1 à esqueda de X
    self.X_ = np.c_[ self.X, np.ones(self.m)]

    # faz one hot enconding de y
    y_ = np.zeros((self.m, self.K))
    for i in range(0, self.m):
      for k in range(0,self.K):
        y_[i, k] = self.y[i]==self.classes[k]

    # Converte de bool para int
    self.y_ = y_.astype(int)

  def soft_max(self, M):
    return np.e**(M)/(np.e**(M)).sum(axis=1).reshape(-1,1)

  def norm_matrix(self,M):
    return np.linalg.norm(M.reshape(-1,1))
