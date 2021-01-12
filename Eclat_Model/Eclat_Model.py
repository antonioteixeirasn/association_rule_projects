#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importando as bibliotecas

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


# Pré-processamento de dados

dataset = pd.read_csv('Market_Basket_Optimisation.csv', header = None)
transactions = [] # cria uma lista
#Criar um ciclo de for para prencher a lista com dados do dataset

for i in range(0, 7501):
  transactions.append([str(dataset.values[i, j]) for j in range(0, 20)])


# In[3]:


# Treinando o Eclat Model no dataset

from apyori import apriori
rules = apriori(transactions = transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2, max_length = 2)


# In[4]:


# Visualizando os resultados

results = list(rules)
results


# In[5]:


# Organizando os resultados em um DataFrame

def inspect(results):
    lhs         = [tuple(result[2][0][0])[0] for result in results]
    rhs         = [tuple(result[2][0][1])[0] for result in results]
    supports    = [result[1] for result in results]
    return list(zip(lhs, rhs, supports))
resultsinDataFrame = pd.DataFrame(inspect(results), columns = ['Product 1', 'Product 2', 'Support'])


# In[6]:


# Visualizando os resultados do DataFrame

resultsinDataFrame.nlargest(n = 10, columns = 'Support')

