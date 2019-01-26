# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 13:09:08 2018

@author: markus
"""

import numpy as np
import pandas as pd
import random
import time
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt


def k_means(x, k, trace = True, maxiter = 10, change = 0.001):
    
    data = pd.DataFrame(x)                                                  # ulozenie datovej vstupnej premennej x do premennej data a zabezbecenie aby bola formatu data.frame
    is_duplicated = True                                                    # inicializacia logickej premenne, aby zbehla prva iteracia vo while-cykle
    
    while is_duplicated:
        index  =  random.sample(list(range(1, data.shape[0] + 1)), k)       # vyber zaciatocnych k riadkov (nahodny vyber k riadkov z dat)
        center_points  = data.iloc[index,:]                                 # ulozenie centier do tabulky center_points
        is_duplicated  = center_points.duplicated().any()                   # kontrola duplicit: ak tam nebude ziadna duplicita, tak funkcia any() vrati FALSE. Funkcia any() vrati TRUE iba ak aspon jedna hodnota bude vo funkcii TRUE (t.j. aspon jedna duplicita)
        center_points["group"] = list(range(1,k + 1))                       # ocislovanie k-skupin (pridanie stlpca do tabulky)
     
    # Inicializacia premennych 
    iter_number = 1              
    small_change = False
    dist_all = float('inf')
  
    
    while(small_change == False and iter_number < (maxiter + 1)): 
        for i in range(k): 
            data["x1"] = center_points.iloc[i, 0]
            data["x2"] = center_points.iloc[i, 1]
            data["group"+str(i+1)] = np.sqrt((data.iloc[:,0]-data["x1"])**2 + (data.iloc[:,1]-data["x2"])**2)  # vypocet euklidovskych distancii od vsetkych centier 
           
                
        data["new_group"] = data[["group" + str(i+1) for i in range(k)]].apply(np.argmin, axis = 1)            # urcenie centra, ktory ma najmensiu euklidovsku vzdialenost
        data["min_distance"] = data[["group" + str(i+1) for i in range(k)]].min(axis = 1)                      # urcenie minimalnej vzdialenosti
  
        dist_all_before = dist_all                 # inicializacia suctu vsetkych vzdialenosti
        dist_all  =  np.sum(data["min_distance"])  # vypocet noveho suctu vsetkych vzdialenosti
  
        # Vizualizacia
        if trace:
            ax = data.plot(kind="scatter", x = data.iloc[:,0].name, y = data.iloc[:,1].name, c = [ int(x[-1:]) for x in  data.new_group],cmap='tab20', colorbar = False)
            center_points.plot(kind="scatter", x = center_points.iloc[:,0].name, y = center_points.iloc[:,1].name, s = 75, marker = "x", c = "black", ax = ax)
            plt.xlabel(data.iloc[:,0].name)
            plt.ylabel(data.iloc[:,1].name)
            plt.title('Iteration n.{}'.format(iter_number))   
            plt.show()
            #plt.clf()
                
            time.sleep(0.5) # toto sposobuje spomalenie procesu, aby to potom vyzeralo, ako animacia ... v pythone to nevyzera moc ako animacia tak to mozes ked tak aj zakomentovat :)  
    
      # Vypocet novych centier (priemerna hodnota suradnic po skupinach)
        center_points = data.groupby(by = data.new_group).mean().iloc[:,:2]   
        center_points["group"] = list(range(1,k+1))
      
        iter_number = iter_number + 1     # navysenie poctu iteracii o 1
        small_change = np.abs((dist_all_before - dist_all))/dist_all < change  # kontrola podmienky na dalsiu iteraciu
    
    ### Output: vyber si jedno z tychto dvoch :), print je podla mna krajsi. Druhe treba zakomentovat 
    
    print("\n", "Number of interations: ", iter_number - 1 , "\n","\n", center_points, "\n","\n", "Grouping: ", data.new_group.values, "\n","\n", "Distances: ", data.min_distance.values)
    #return(iter_number,center_points, data.new_group.values, data.min_distance.values)

# Testovanie na simulovanych datach

samp1 = np.random.multivariate_normal([1, 1], np.identity(2), 25)
samp2 = np.random.multivariate_normal([-1, 1], np.identity(2), 25)
samp3 = np.random.multivariate_normal([1, -1], np.identity(2), 25)
samp4 = np.random.multivariate_normal([-1, -1], np.identity(2), 25)
samp = np.concatenate([samp1,samp2,samp3,samp4])

k_means(x = samp, k = 4, trace = True, maxiter = 10, change = 0.001)


# Testovanie na umelych clustroch pomocou funkcie make_blobs

data, blob_num = make_blobs(n_samples=500, centers=3, n_features=2, random_state= 2018 )

k_means(x = data, k = 3, trace = True, maxiter = 10, change = 0.001)
k_means(x = data, k = 5, trace = True, maxiter = 10, change = 0.001)





