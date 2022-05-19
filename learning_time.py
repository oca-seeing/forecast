#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 27 15:23:27 2020

@author: cgiordano
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  9 10:14:36 2020

@author: cgiordano
"""
import sys
sys.path.insert(0,'../')
from sklearn import linear_model
from sklearn.impute import SimpleImputer
from sklearn import svm
from sklearn.ensemble import RandomForestRegressor
from Data_preparation import data_preparation_time
import numpy as np
import matplotlib.pyplot as plt

"""
Data preparation
"""
print('Data preparation')
xtraining, ytraining, xtest, ytest, xkey, ykey, ytrainingisop, ytestisop, ytrainingtau0, ytesttau0 = data_preparation_time.data_preparation()

"""
Machine learning 
Use of 2 algorythms:
    - Ridge
    - Random Forest
"""

print('Initialize regressors')
reg = linear_model.Ridge(alpha=2.0)
clf = svm.SVR(epsilon=0.001, cache_size=2000)
rdmfr = RandomForestRegressor(n_estimators=100)

"""        
Data preparation to remove missing values
"""
print('Remove missing values')
imp = SimpleImputer(missing_values=np.nan, strategy='mean')

print('    X training seeing')
imp.fit(xtraining)
xtraining = imp.transform(xtraining)
print('    Y training seeing')
imp.fit(ytraining)
ytraining = imp.transform(ytraining)
print('    Y training isop')
imp.fit(ytrainingisop)
ytrainingisop = imp.transform(ytrainingisop)
print('    Y training tau0')
imp.fit(ytrainingtau0)
ytrainingtau0 = imp.transform(ytrainingtau0)
print()

print('    X test seeing')
imp.fit(xtest)
xtest = imp.transform(xtest)
print('    Y test seeing')
imp.fit(ytest)
ytestclean = imp.transform(ytest)#clean ytest
print('    Y test isop')
imp.fit(ytestisop)
ytestisopclean = imp.transform(ytestisop)#clean ytest
print('    Y test tau0')
imp.fit(ytesttau0)
ytesttau0clean = imp.transform(ytesttau0)#clean ytest

"""
Regression on training data
prediction on test data
"""

print('Ridge regression')
reg.fit(xtraining, ytraining)
ypredict = reg.predict(xtest)

print('Ridge regression isop')
reg.fit(xtraining, ytrainingisop)
ypredictisop = reg.predict(xtest)

print('Ridge regression tau0')
reg.fit(xtraining, ytrainingtau0)
ypredicttau0 = reg.predict(xtest)

print('Random Forest Regression')
rdmfr.fit(xtraining, ytraining)
print('prediction')
ypredict_rdmfr = rdmfr.predict(xtest)

print('Random Forest Regression isop')
rdmfr.fit(xtraining, ytrainingisop)
ypredictisop_rdmfr = rdmfr.predict(xtest)

print('Random Forest Regression tau0')
rdmfr.fit(xtraining, ytrainingtau0)
ypredicttau0_rdmfr = rdmfr.predict(xtest)

#print('SVM')
#ypredict_cfl = []
#for i in np.arange(4):
 #   print(i)
  #  ypredict_cfl.append(clf.fit(xmatrix, ymatrix[:,i]).predict(xmatrix_test))

print('prediction')

# for i in  np.arange(4):
#     plt.scatter(ypredict[:,i],ymatrix_test[:,i],s=1, label="ridge")
#     plt.xlabel("predictions")
#     plt.ylabel("measurements")
#     plt.title(ykey[i])
#     plt.xlim((np.nanmin(ymatrix_test[:,i]), np.nanmax(ymatrix_test[:,i])))
#   # plt.scatter(ypredict_cfl[i],ymatrix_test[:,i],s=1, label="svm",alpha=0.05)
#     plt.scatter(ypredict_rdmfr[:,i],ymatrix_test[:,i],s=1, label="Random F", alpha=0.1)
#     plt.legend()
#     plt.show()

"""
Plot results in scatter plot
"""
print('Plot results on seeing predictions')
plt.scatter(ypredict,ytest,s=1, label="ridge")
plt.xlabel("Predictions")
plt.ylabel("Measurements")
plt.title(ykey[0]+' [arcsec]')
plt.xlim((np.nanmin(ytest), np.nanmax(ytest)))
# plt.scatter(ypredict_cfl[i],ymatrix_test[:,i],s=1, label="svm",alpha=0.05)
plt.scatter(ypredict_rdmfr,ytest,s=1, label="Random F", alpha=0.5)
plt.plot(np.array([0,4]),np.array([0,4]),'r-')
plt.axis('square')
plt.axis([0,3,0,3])
plt.xticks(np.arange(0.,3.1,0.5))
plt.yticks(np.arange(0.,3.1,0.5))
plt.legend()
plt.show()

corr_mat = np.corrcoef(ypredict, ytestclean, rowvar=False)
corr_mat_rdmfr = np.corrcoef(ypredict_rdmfr, ytestclean, rowvar=False)

#df = pd.Dataframe()
corr = [corr_mat[i,i+24] for i in np.arange(24)]
corr_rdmfr = [corr_mat_rdmfr[i,i+24] for i in np.arange(24)]

plt.plot(range(0,120,5),corr, label="ridge", linewidth=2)
plt.plot(range(0,120,5),corr_rdmfr, label="random F",linewidth=2)
plt.xlabel("Minutes predicted")
plt.ylabel("Correlation coefficient")
plt.axis([0,120,0,1])
plt.legend()
plt.show()

print('Plot results on isoplanatic angle predictions')
plt.scatter(ypredictisop,ytestisop,s=1, label="ridge")
plt.xlabel("Predictions")
plt.ylabel("Measurements")
plt.title('Isop [arcsec]')
plt.xlim((np.nanmin(ytest), np.nanmax(ytest)))
# plt.scatter(ypredict_cfl[i],ymatrix_test[:,i],s=1, label="svm",alpha=0.05)
plt.scatter(ypredictisop_rdmfr,ytestisop,s=1, label="Random F", alpha=0.5)
plt.plot(np.array([0,4]),np.array([0,4]),'r-')
plt.axis('square')
plt.axis([0,3,0,3])
plt.xticks(np.arange(0.,3.1,0.5))
plt.yticks(np.arange(0.,3.1,0.5))
plt.legend()
plt.show()

corr_mat = np.corrcoef(ypredictisop, ytestisopclean, rowvar=False)
corr_mat_rdmfr = np.corrcoef(ypredictisop_rdmfr, ytestisopclean, rowvar=False)

#df = pd.Dataframe()
corr = [corr_mat[i,i+24] for i in np.arange(24)]
corr_rdmfr = [corr_mat_rdmfr[i,i+24] for i in np.arange(24)]

plt.plot(range(0,120,5),corr, label="ridge", linewidth=2)
plt.plot(range(0,120,5),corr_rdmfr, label="random F",linewidth=2)
plt.xlabel("Minutes predicted")
plt.ylabel("Correlation coefficient")
plt.axis([0,120,0,1])
plt.legend()
plt.show()

print('Plot results on coherence time predictions')
plt.scatter(ypredicttau0,ytesttau0,s=1, label="ridge")
plt.xlabel("Predictions")
plt.ylabel("Measurements")
plt.title('tau0 [arcsec]')
plt.xlim((np.nanmin(ytest), np.nanmax(ytest)))
# plt.scatter(ypredict_cfl[i],ymatrix_test[:,i],s=1, label="svm",alpha=0.05)
plt.scatter(ypredicttau0_rdmfr,ytesttau0,s=1, label="Random F", alpha=0.5)
plt.plot(np.array([0,4]),np.array([0,4]),'r-')
plt.axis('square')
plt.axis([0,3,0,3])
plt.xticks(np.arange(0.,3.1,0.5))
plt.yticks(np.arange(0.,3.1,0.5))
plt.legend()
plt.show()

corr_mat = np.corrcoef(ypredicttau0, ytesttau0clean, rowvar=False)
corr_mat_rdmfr = np.corrcoef(ypredicttau0_rdmfr, ytesttau0clean, rowvar=False)

#df = pd.Dataframe()
corr = [corr_mat[i,i+24] for i in np.arange(24)]
corr_rdmfr = [corr_mat_rdmfr[i,i+24] for i in np.arange(24)]

plt.plot(range(0,120,5),corr, label="ridge", linewidth=2)
plt.plot(range(0,120,5),corr_rdmfr, label="random F",linewidth=2)
plt.xlabel("Minutes predicted")
plt.ylabel("Correlation coefficient")
plt.axis([0,120,0,1])
plt.legend()
plt.show()

plt.plot(range(0,120,5),ypredict_rdmfr[125,:], label="Prediction", linewidth=2)
plt.plot(range(0,120,5),ytestclean[125,:], 'r-*', label="Real measurements",linewidth=2)
plt.xlabel("Minutes")
plt.ylabel("Seeing in arcsec")
plt.axis([0,120,0,2])
plt.legend()
plt.show()

print()
