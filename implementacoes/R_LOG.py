# -*- coding: utf-8 -*-
import numpy as np
from numpy.random import seed

def softmax(Xtr, w, instancias, classes):
  S = np.zeros((instancias, classes))
  Y = np.zeros((instancias, classes))
  S = np.dot(Xtr, w.T) 
  
  for i in range(instancias):
    Y[i] = np.exp(S[i]) / (np.sum(np.exp(S[i]) * np.ones((1,classes))))
  
  return Y

def h_linha(alfa, w, gradiente, X, y, instancias, classes):
  w = w - alfa * gradiente
  Y = softmax(X, w, instancias, classes)
  grad_a = np.dot((Y - y).T, X) 
 
  return  np.dot(grad_a.flatten(), (-1*gradiente.flatten().T))

def bissecao(w, gradiente, X, y, instancias, classes):
    al = 0.0
    seed(42)
    au = np.random.rand() #0.0001 # Chute inicial
    it = 0

    hl = h_linha(au, w, gradiente, X, y,instancias, classes)

    while hl < 0: #Verifica au está acima de zero.
      al = au
      au = au * 2
      hl = h_linha(au, w, gradiente, X, y,instancias, classes)
      
    a_medio = (al + au) / 2

    hl = h_linha(a_medio, w, gradiente, X, y,instancias, classes)

    itmax = np.int(np.ceil(np.log(au-al) - np.log(1e-5))/np.log(2)) #interações.

    
    #while (it < itmax) and ((au - al)<=1.0) and (np.abs(hl)>10.0):
    while (it < itmax):
      it += 1
      if hl > 0:
        au = a_medio
      elif hl < 0:
        al = a_medio
      elif hl == 0:
        return a_medio
      
      a_medio = (al + au) / 2
      hl = h_linha(a_medio, w, gradiente, X, y, instancias, classes)

    return a_medio

def regressao_logistica_bicessao(X, y, itmax= 2500, taxa_aprendizado_fixa = None, verbose= True, arquivo=None):
  '''
    Calcula a regressão Logística multivariada com bicessap
    X = Matriz de dados de entrada com bias;
    y = vetor de saída.
    ___________
    return: o valor de w para predição.
  '''
  instancias, atributos = np.shape(X)
  classes =  y.shape[1]
  seed(42)
  w = np.random.rand(classes,atributos)
  Y = softmax(X, w, instancias, classes)
  erro = (Y - y)
  E = Entropia_Cruzada(y, Y, instancias)

  gradiente = np.dot(erro.T, X)

  alfa = 0.1
  #itmax= 5000
  it = 0

  E_ant = 0
  E_atual = 0
  soma_E = 0

  print("Interação: ",it, "E:",  E)
 
  #gradiente_norm = np.linalg.norm(gradiente.flatten()) #para testar até 10000

  while (it < itmax) and (soma_E < 10): #and (gradiente_norm > 1e-5): #and (it < itmax):
    it += 1

    if taxa_aprendizado_fixa is None:
      alfa = bissecao(w,gradiente, X, y, instancias, classes) # alfa variável
    else:
      alfa = taxa_aprendizado_fixa
    
    w = w - alfa * gradiente # atualiza os pesos com gradiente
    
    Y = softmax(X, w, instancias, classes)
    erro = (Y - y)
    gradiente = np.dot(erro.T, X)
    E = Entropia_Cruzada(y, Y, instancias)

    #gradiente_norm = np.linalg.norm(gradiente.flatten())

    if ((it % 100)==0):
      if verbose:
        print("Interação: ",it, "E:",  E, "alfa:", alfa, file=arquivo)
      print("Interação: ",it, "E:",  E, "alfa:", alfa)

    if E_ant == E: #Conta a quantidade de MSE iguais
      if E_ant == E_atual:
        soma_MSE +=1
      else:
        soma_MSE = 1
        E_atual = E
    else:  
      E_ant = E
  
  return w

def Entropia_Cruzada(Ytr, Y, instancias):
  soma = 0
  for i in range(instancias):
    soma += np.sum(np.dot(Ytr[i],np.log(Y[i]).T)) 
  return (-1*soma)

def preditor_logistico(X,w, instancias, classes):
  '''
    Calcula as predições para entradas
    X = Matriz de dados de entrada
    w = pesos gerados pelo gradiente
    ________
    return: o vetor de resultado. 
  '''
  S = np.zeros((instancias, classes))

  Y = np.zeros((instancias, classes))

  S = np.dot(X, w.T)
  Y = softmax(X, w, instancias, classes)

  Y = Y.argmax(axis=1)

  return Y