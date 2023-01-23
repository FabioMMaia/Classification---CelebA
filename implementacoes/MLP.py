# -*- coding: utf-8 -*-


from numpy.random import seed
from sklearn.metrics import accuracy_score
import numpy as np


def bissecao_rna(dEdA, dEdB, A, B, X, Yd, N):
    
    def h_linha(au, dEdA, dEdB, dv, X, Yd, N):
      An = A - au*dEdA;
      Bn = B - au*dEdB;

      dJdAn, dJdBn = calculo_gradiente(X, Yd, An, Bn, N)
      g=np.array((dJdAn.flatten(),dJdBn.flatten()))
      
      return  (np.sum(dv[0] * g[0]) + np.sum(dv[1] * g[1]))
    
    dv = np.array((-dEdA.flatten(),-dEdB.flatten())) 
    
    au = np.random.rand() # Chute inicial

    al = 0.0
    au = np.random.rand() # Chute inicial
    it = 0

    hl = h_linha(au, dEdA, dEdB, dv, X, Yd, N)

    while hl < 0: #Verifica au está acima de zero.
      al = au
      au = au * 2
      hl = h_linha(au, dEdA, dEdB, dv, X, Yd, N)
      
    a_medio = (al + au) / 2

    hl = h_linha(au, dEdA, dEdB, dv, X, Yd, N)

    itmax = np.int(np.ceil(np.log(au-al) - np.log(1e-5))/np.log(2)) 

    while (it < itmax):
      it += 1
      if hl > 0:
        au = a_medio
      elif hl < 0:
        al = a_medio
      elif hl == 0:
        return a_medio
      
      a_medio = (al + au) / 2
      hl = h_linha(au, dEdA, dEdB, dv, X, Yd, N)

    return a_medio


def calculo_saida(X, A, B, N):
  '''
    Calcula as predições para entradas usando argmax()
    X = Entrada da rede.
    A = Pesos da rede da camada do meio.
    b = Pesos da rede da camada de sáida.
    N = Número de instâncias do X.
    ________
    return: o vetor de saída com os valores das classes.
  '''
  Zin = np.dot(X, A.T)
  Z = 1/(1+np.exp(-Zin))

  Z = np.column_stack((Z,np.ones(N)))
  Yin = np.dot(Z, B.T)
  Y = 1/(1+np.exp(-Yin))

  return Y

def calculo_gradiente(X, Yd, A, B, N):
  Zin = np.dot(X, A.T) 
  Z = 1/(1+np.exp(-Zin)) 

  Zb = np.column_stack((Z,np.ones(N))) 
  Yin = np.dot(Zb, B.T) 
  Y = 1/(1+np.exp(-Yin)) 

  erro = Y -Yd 
  dg = (1-Y)*Y 
  df = (1-Z)*Z 
  

  dEdB = 1/N * np.dot((erro * dg).T, Zb) 

  dEdZ = np.dot((erro* dg), B[:,:-1]) 
  

  dEdA = 1/N * np.dot((dEdZ* df).T, X)  

  return dEdA, dEdB

def preditor_sigmoide(Y):
  '''
    Calcula as predições para entradas usando argmax()
    Y = relação com as saídas
    ________
    return: o vetor do resultado. 
  '''
  Y = Y.argmax(axis=1)
  return Y

def rna(X, Yd, h, taxa_aprendizado_fixa=None, verbose = True, epoca = 3000, arquivo='resultadoFile'):
  N, ne = X.shape
  ns = Yd.shape[1]

  X = np.column_stack((X,np.ones(N)))
  seed(42)
  A = np.random.rand(h,(ne+1)) / 2  
  B = np.random.rand(ns,(h+1)) / 2  

  Y = calculo_saida(X, A, B, N)
  erro = Y - Yd


  MSE = 1/N * np.sum(erro*erro)
  
  ep = 0
  epmax = epoca
  E = 1e-2
  MSE_ant = 0
  MSE_atual = 0
  soma_MSE = 0
  
  a,b = [],[]# Utilizada no Gráfico.

  #g =  np.concatenate([A.flatten(), B.flatten()])

  gradiente_norm = 1

  while (ep<epmax) and (soma_MSE < 20): #and (np.linalg.norm(g)>1e-7):
    ep += 1
    dEdA, dEdB = calculo_gradiente(X, Yd, A, B, N)

    if taxa_aprendizado_fixa is None:
      alfa = bissecao_rna(dEdA, dEdB, A, B, X, Yd, N)
    else:
      alfa = taxa_aprendizado_fixa

    A = A - alfa*dEdA
    B = B - alfa*dEdB

    #g =  np.concatenate([dEdA.flatten(), dEdB.flatten()])

    Y = calculo_saida(X, A, B, N)
    erro = Y - Yd
    
    
    MSE = 1/N * np.sum(erro*erro)

    if ((ep % 100)==0):
      if verbose:
        print("Parâmetro :", h, "Epocas: ",ep, "MSE:", MSE, "ALFA:", alfa, 
          "Acurácia:",round(accuracy_score(preditor_sigmoide(Y), preditor_sigmoide(Yd)),2), file=arquivo)
        
      print("Parâmetro :", h, "Epocas: ",ep, "MSE:", MSE, "ALFA:", alfa, 
          "Acurácia:",round(accuracy_score(preditor_sigmoide(Y), preditor_sigmoide(Yd)),2))
    if verbose:
      a.append(ep)
      b.append(MSE)

    if MSE_ant == MSE: #Conta a quantidade de MSE iguais
      if MSE_ant == MSE_atual:
        soma_MSE +=1
      else:
        soma_MSE = 1
        MSE_atual = MSE
    else:  
      MSE_ant = MSE

  return A,B