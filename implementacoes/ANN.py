# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

"""# Implementação da Classe RNA"""

### RNA com alpha fixo ###

class RNA():

  def __init__(self,h=5, nepocasmax = 50000, alfa=0.1, early_stop=False):
    self.h = h
    self.nepocasmax = nepocasmax
    self.alfa = alfa
    self.early_stop=early_stop

  def fit(self, Xtr, Ytr, eval):

    assert isinstance(eval,list) and len(eval)==2, 'eval deve ser uma lista contendo [Xval, Yval]'

    self.Xtr = pd.DataFrame(Xtr).values
    self.Ytr = pd.DataFrame(Ytr).values
    self.Xval = pd.DataFrame(eval[0]).values
    self.Yval = pd.DataFrame(eval[1]).values
    (self.Ntr,self.ne) = pd.DataFrame(Xtr).shape
    (self.Nval,_ )=pd.DataFrame(self.Xval).shape
    self.ns= pd.DataFrame(Ytr).shape[1]
    self.Aot = None
    self.Bot = None

    # adciona o bias 
    self.Xtr = np.c_[ self.Xtr, np.ones(self.Ntr)]
    self.Xval = np.c_[ self.Xval, np.ones(self.Nval)]

    # Gerar as matrizes de pesos
    A= np.random.random((self.h,self.ne + 1))/2
    B= np.random.random((self.ns,self.h + 1))/2

    # calcula a saida
    Y= self.forward(self.Xtr,A,B)
    
    # calcula erro
    erro_tr = Y - self.Ytr

    # Calcula do MSE
    EQM= (erro_tr**2).sum().sum()/self.Ntr

    # Calcular o gradiente
    (dJdA,dJdB)=self.calc_grad(self.Xtr,self.Ytr,A,B,self.Ntr);
    g=  np.concatenate([dJdA.flatten(), dJdB.flatten()])
    nep = 0
    vetErro_tr=[]
    vetErro_val=[]
    # vetErro.append(EQM)

    # faz o forward
    Yv=self.forward(self.Xval,A,B)
    erro_val= Yv-self.Yval
    EQM_val_ant= (erro_val**2).sum().sum()/self.Nval
    cont=0

    # condição de parada
    while np.linalg.norm(g)>1e-5 and nep<self.nepocasmax:
      nep=nep+1

      # gradiente descendente
      A = A - self.alfa*dJdA;
      B = B - self.alfa*dJdB;

      # backward
      (dJdA,dJdB)=self.calc_grad(self.Xtr,self.Ytr,A,B,self.Ntr)
      g=  np.concatenate([dJdA.flatten(), dJdB.flatten()])

      # Calcula erro de treino e validacao
      Y = self.forward(self.Xtr,A,B)
      erro_tr= Y-self.Ytr
      EQM_tr= (erro_tr**2).sum().sum()/self.Ntr
      vetErro_tr.append(EQM_tr)
      Yv=self.forward(self.Xval,A,B)
      erro_val= Yv-self.Yval
      EQM_val= (erro_val**2).sum().sum()/self.Nval
      vetErro_val.append(EQM_val)

      if nep%10000==0:
        print('nep:', nep, 'EQM:', EQM, 'EQM_val:', EQM_val, 'norm:', np.linalg.norm(g))
      
      if EQM_val_ant>EQM_val:
        self.Aot = A
        self.Bot = B
        EQM_val_ant= EQM_val
        cont=0
      else:
        cont+=1
    
      if cont>100 and self.early_stop:
        break

    print('Convergência em {} epochs'.format(nep))
    self.A=A
    self.B=B
    self.iter_conv = nep
    self.vetErro_tr = vetErro_tr
    self.vetErro_val = vetErro_val

    # return Y, Yv

  def forward(self, X, A, B):
    N = X.shape[0]
    Zin = np.dot(X, A.T)
    Z=1./(1+np.exp(-Zin))
    Z= np.c_[ Z, np.ones(N)]
    Yin = np.dot(Z,B.T)
    Y= 1./(1+np.exp(-Yin))
    return Y

  def predict(self,X):
    return np.argmax(self.predict_proba(X),axis=1)

  def predict_proba(self, X):
    N = X.shape[0]
    X = np.c_[ X, np.ones(N)]
    Zin = np.dot(X, self.A.T)
    Z=1./(1+np.exp(-Zin))
    Z= np.c_[ Z, np.ones(N)]
    Yin = np.dot(Z,self.B.T)
    Y= 1./(1+np.exp(-Yin))
    return Y

  def calc_grad(self,Xtr,Ytr,A,B,N):
    Zin = np.dot(Xtr,A.T)
    Z=1./(1+np.exp(-Zin))
    gl=(1-Z)*Z;
    Z= np.c_[ Z, np.ones(N)]
    Yin = np.dot(Z,B.T)
    Y= 1./(1+np.exp(-Yin))
    erro = Y-Ytr
    fl = (1-Y)*Y
    dJdB = 1/N*np.dot((erro*fl).T, Z)
    dJdZ = np.dot((erro*fl),B[:,:-1])
    dJdA = 1/N*np.dot((dJdZ*gl).T, Xtr)
    return (dJdA, dJdB)

### RNA + bissecao ###

class RNA_Bissecao(RNA):

  def __init__(self,h=5, nepocasmax = 50000, early_stop=False, epsilon_bissec=1e-5, itmax_bissec=50):
    super().__init__(h,nepocasmax, early_stop)
    self.epsilon_bissec=epsilon_bissec
    self.itmax_bissec = itmax_bissec

  def fit(self, Xtr, Ytr, eval):

    assert isinstance(eval,list) and len(eval)==2, 'eval deve ser uma lista contendo [Xval, Yval]'
    self.Xtr = pd.DataFrame(Xtr).values
    self.Ytr = pd.DataFrame(Ytr).values
    self.Xval = pd.DataFrame(eval[0]).values
    self.Yval = pd.DataFrame(eval[1]).values
    (self.Ntr,self.ne) = pd.DataFrame(Xtr).shape
    (self.Nval,_ )=pd.DataFrame(self.Xval).shape
    self.ns= pd.DataFrame(Ytr).shape[1]
    self.Aot = None
    self.Bot = None

    # adciona o bias 
    self.Xtr = np.c_[ self.Xtr, np.ones(self.Ntr)]
    self.Xval = np.c_[ self.Xval, np.ones(self.Nval)]

    # Gerar as matrizes de pesos
    A= np.random.random((self.h,self.ne + 1))/2
    B= np.random.random((self.ns,self.h + 1))/2

    # calcula a saida
    Y= self.forward(self.Xtr,A,B)
    
    # calcula erro
    erro_tr = Y - self.Ytr

    # Calcula do MSE
    EQM= (erro_tr**2).sum().sum()/self.Ntr

    # Calcular o gradiente
    (dJdA,dJdB)=self.calc_grad(self.Xtr,self.Ytr,A,B,self.Ntr);
    g=  np.concatenate([dJdA.flatten(), dJdB.flatten()])
    nep = 0
    vetErro_tr=[]
    vetErro_val=[]
    # vetErro_tr.append(EQM)

    # faz o forward
    Yv=self.forward(self.Xval,A,B)
    erro_val= Yv-self.Yval
    EQM_val_ant= (erro_val**2).sum().sum()/self.Nval
    cont=0

    # condição de parada
    while np.linalg.norm(g)>1e-4 and nep<self.nepocasmax:
      nep=nep+1

      # gradiente descendente
      alfa = self.calculate_alfa(dJdA,dJdB,A,B)
      A = A - alfa*dJdA;
      B = B - alfa*dJdB;

      # backward
      (dJdA,dJdB)=self.calc_grad(self.Xtr,self.Ytr,A,B,self.Ntr)
      g=  np.concatenate([dJdA.flatten(), dJdB.flatten()])

      # Calcula erro de treino e validacao
      Y = self.forward(self.Xtr,A,B)
      erro_tr= Y-Ytr
      EQM= (erro_tr**2).sum().sum()/self.Ntr
      vetErro_tr.append(EQM)
      Yv=self.forward(self.Xval,A,B)
      erro_val= Yv-self.Yval
      EQM_val= (erro_val**2).sum().sum()/self.Nval
      vetErro_val.append(EQM_val)

      if nep%10000==0:
        print('nep:', nep, 'EQM:', EQM, 'EQM_val:', EQM_val, 'norm:', np.linalg.norm(g), 'alfa:', alfa)
      
      if EQM_val_ant>EQM_val:
        self.Aot = A
        self.Bot = B
        EQM_val_ant= EQM_val
        cont=0
      else:
        cont+=1
    
      if cont>100 and self.early_stop:
        break

    # print('Convergência em {} epochs'.format(nep))
    self.iter_conv = nep
    self.vetErro_tr = vetErro_tr
    self.vetErro_val = vetErro_val
    self.A=A
    self.B=B

    # return Y, Yv

  def calculate_alfa(self, dJdA,dJdB,A,B):
    
    dv = np.concatenate([-dJdA.flatten() ,-dJdB.flatten()])
    alfa_u = np.random.random()
    An = A - alfa_u*dJdA
    Bn = B - alfa_u*dJdB

    (dJdAn,dJdBn)=self.calc_grad(self.Xtr,self.Ytr,An,Bn,self.Ntr)
    g = np.concatenate([dJdAn.flatten() , dJdBn.flatten()])
    hl = np.dot(g.T, dv)

    alfa_l=0
    while hl<0:
      alfa_l = alfa_u
      alfa_u = alfa_u*2
      An = A - alfa_u*dJdA
      Bn = B - alfa_u*dJdB
      (dJdAn,dJdBn)=self.calc_grad(self.Xtr,self.Ytr,An,Bn,self.Ntr)
      g = np.concatenate([dJdAn.flatten() , dJdBn.flatten()])
      hl = np.dot(g.T, dv)

    kmax = np.ceil(np.log2((alfa_u-alfa_l)/self.epsilon_bissec))
    it=0

    while it<kmax and it<self.itmax_bissec and abs(hl)>self.epsilon_bissec:
      # print('kmax:', kmax, 'it:',it, 'alfa_m:',  (alfa_l + alfa_u)/2, 'hl:', hl)
      it+=1
      alfa_m = (alfa_l + alfa_u)/2
      An = A - alfa_u*dJdA
      Bn = B - alfa_u*dJdB
      (dJdAn,dJdBn)=self.calc_grad(self.Xtr,self.Ytr,An,Bn,self.Ntr)
      g = np.concatenate([dJdAn.flatten() , dJdBn.flatten()])
      hl = np.dot(g.T, dv)

      if abs(hl)<1e-3:
        break
      elif hl>0:
        alfa_u = alfa_m
      elif hl <0:
        alfa_l = alfa_m
    

    alfa_m = (alfa_l + alfa_u)/2

    return alfa_m
