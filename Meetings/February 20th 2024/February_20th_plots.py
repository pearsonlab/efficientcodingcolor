# -*- coding: utf-8 -*-
"""
Created on Tue Feb 20 12:15:11 2024

@author: David
"""
plt.figure()
plt.xlim(0,5)
plt.hist(test1.a, bins = 20)

plt.plot(test1.b[0,:], test2.b[0,:], 'o')
plt.xlabel("Surround precision (uncorrelated)", size = 30)
plt.ylabel("Surround precision (correlated)", size = 30)

test1.a[0,1/test1.a[0,:] > 100] = 1
test2.a[0,1/test2.a[0,:] > 100] = 1

sns.histplot(1/test1.a[0,test1.type == 0], bins = 100)
sns.histplot(1/test1.a[0,test1.type == 1], bins = 100)
plt.figure()
sns.histplot(1/test2.a[0,test2.type == 0], bins = 100)
sns.histplot(1/test2.a[0,test2.type == 1], bins = 100)
plt.figure()
sns.histplot(1/test1.a[0,:], binwidth = 0.2)
sns.histplot(1/test2.a[0,:], binwidth = 0.2)
sns.histplot(test1.gauss_params[:,3], binwidth = 0.2)
sns.histplot(test2.gauss_params[:,3], binwidth = 0.2)
#sns.histplot(1/test2.a[0,:], bins = 30)
plt.xlabel("Gaussian fit SD", size = 50)
plt.ylabel("Number of neurons", size = 50)
plt.xticks(size = 35)
plt.yticks(size=35)
plt.title("DoG fits to unparametrized RFs", size = 50)

plt.figure()
sns.histplot(test1.c[0,:], bins = 100)
sns.histplot(test2.c[0,:], bins = 100)