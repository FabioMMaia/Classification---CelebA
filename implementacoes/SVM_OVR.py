# from modules import SVM
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
import cvxopt
from tqdm import tqdm

class SVM():

  def __init__(self, C, kernel_type='poly', kernel_params=None, solver ="quadprog", tol=1e-5, verbose=False ):
    assert C>0, 'C precisa ser uma constante positiva'
    self.C=float(C)
    self.kernel_type = kernel_type
    self.kernel_params = kernel_params
    self.solver = solver
    self.tol = tol
    self.verbose=verbose
    cvxopt.solvers.options['show_progress'] = verbose

  def fit(self,X,y):

    class_labels = np.unique(y)
    assert len(class_labels)==2, "y precisa ser binário"

    X = np.array(X)
    y = np.array(pd.DataFrame(y))
    self.m = X.shape[0]

    self.X=X
    self.y=y
    
    # Equação principal
    P = self.calculate_kernel()

    q = -np.ones((self.m,1)).reshape(-1,1)

    # Inegualidades
    G_lb = -np.eye(self.m)
    h_lb = np.zeros(self.m).reshape(-1,1)

    G_ub = np.eye(self.m)
    h_ub = self.C*np.ones(self.m).reshape(-1,1)

    G = np.vstack((G_lb,G_ub))
    h = np.vstack((h_lb,h_ub))

    # Igualdade
    A = y.reshape(1,-1).astype(float)
    b = 0.0

    P       = cvxopt.matrix(P)
    q       = cvxopt.matrix(q)
    G       = cvxopt.matrix(G)
    h       = cvxopt.matrix(h)
    A       = cvxopt.matrix(A)
    b       = cvxopt.matrix(b)

    solved  = cvxopt.solvers.qp(P, q, G, h, A, b, show_progress=self.verbose) ;
    alpha = solved['x']

    alpha = np.array(alpha)
    sv = (alpha>1e-5).flatten()
    self.alpha = alpha
    self.sv = sv

  def check_param(self, param):
    assert param in self.kernel_params.keys(), 'se o kernel é {} é necessário ter declarado o parametro {}'.format(self.kernel_type,param)

  def calculate_kernel(self):

    assert isinstance(self.kernel_params, dict), 'kernel_params precisa ser um dicionario'
    assert self.kernel_type in ['poly', 'rbf'], 'kernel_type precisa ser poly ou rbf'

    K = np.zeros((self.m, self.m))
    H = np.zeros((self.m, self.m))

    if self.kernel_type=='poly':
      self.check_param('d')
      d = self.kernel_params['d']

      for i in range(0, self.m):
        for j in range(0, self.m):
          K[i,j] = (self.X[i,:]@self.X[j,:]+1)**d
          H[i,j] = K[i,j]*self.y[i,0]*self.y[j,0]

      return H

    elif self.kernel_type=='rbf':
      self.check_param('sigma')
      sigma = self.kernel_params['sigma']

      for i in range(0, self.m):
        for j in range(0, self.m):
          K[i,j] = np.exp(-np.linalg.norm(self.X[i,:] - self.X[j,:])**2/(2*sigma**2))
          H[i,j] = K[i,j]*self.y[i,0]*self.y[j,0]
          
      return H
      

  def predict(self, X_tst, confidence=False):

    X_tst = np.array(X_tst)
    m_test = X_tst.shape[0]

    alphasv = self.alpha[self.sv]
    Xsv = self.X[self.sv,:]
    ysv = self.y[self.sv,:]
    m_sv = Xsv.shape[0]

    K = np.zeros((m_sv, m_test))

    for i in range(0, m_sv):
      for j in range(0, m_test):
        if self.kernel_type=='poly':
          K[i,j] = (Xsv[i,:] @ X_tst[j,:]+1)**self.kernel_params['d']

        elif self.kernel_type=='rbf':
          K[i,j] = np.exp(-np.linalg.norm(Xsv[i,:].flatten() - X_tst[j,:].flatten())**2/(2*self.kernel_params['sigma']**2))
          
    yhat = np.sign((alphasv * ysv).T@K)

    if confidence:
      return (yhat, (alphasv * ysv).T@K)
    else:
      return yhat


class SVM_one_vs_all():

    def __init__(self,C=1, sigma=0.1, n_rounds = None, kernel_type='rbf', m_value=8):
        self.C=C
        self.sigma=sigma
        self.kernel_type = kernel_type
        self.n_rounds=n_rounds
        self.m_value=m_value


    def fit(self, X_train, y_train):
        X_train = np.array(X_train)
        y_train = np.array(pd.DataFrame(y_train))
        self.n_classes = len(np.unique(y_train))

        assert self.n_classes>2, 'Necessário se ter mais de duas classe na base de treino'

        models = {}

        for classe in tqdm(np.unique(y_train)):
            y_train_trns_bool = y_train==classe
            y_train_trns = y_train_trns_bool.astype(int)*2 - 1 

            classifier = SVM(C=self.C, kernel_type=self.kernel_type, kernel_params={'sigma':self.sigma}, verbose=False)
            
            X_train_pos = X_train[y_train_trns_bool.flatten(),:]
            m_pos = X_train_pos.shape[0]
            X_train_neg = X_train[~y_train_trns_bool.flatten()][:self.m_value*m_pos,:]
            y_train_trns_pos = y_train_trns[y_train_trns_bool.flatten()]
            y_train_trns_neg = y_train_trns[~y_train_trns_bool.flatten()][:self.m_value*m_pos]
            
            X_train_f = np.vstack((X_train_pos,X_train_neg ))
            y_train_f = np.hstack((y_train_trns_pos.flatten(),y_train_trns_neg.flatten()))

            classifier.fit(X_train_f, y_train_f.reshape(-1,1))
            
            models[classe]= classifier

        self.models=models

    def predict(self, X):

        if isinstance(X,pd.DataFrame):
            X = np.array(X)

        outputs = []
        confidences = []

        for i,(class_n, model) in enumerate(self.models.items()):

          (label, conf) = model.predict(X, confidence=True)

          if i==0:
            confidences = conf.reshape(-1,1)
            outputs = label.reshape(-1,1)

          else:
            confidences = np.hstack((confidences, conf.reshape(-1,1)))
            outputs = np.hstack((outputs, label.reshape(-1,1)))
        
        self.confidences=confidences
        return self.confidences.argmax(axis=1)


    def predict_proba(self, X):
      self.predict(X)
      return self.confidences






