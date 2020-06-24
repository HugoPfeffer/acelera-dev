#!/usr/bin/env python
# coding: utf-8

# # Desafio 1
# 
# Para esse desafio, vamos trabalhar com o data set [Black Friday](https://www.kaggle.com/mehdidag/black-friday), que reúne dados sobre transações de compras em uma loja de varejo.
# 
# Vamos utilizá-lo para praticar a exploração de data sets utilizando pandas. Você pode fazer toda análise neste mesmo notebook, mas as resposta devem estar nos locais indicados.
# 
# > Obs.: Por favor, não modifique o nome das funções de resposta.

# ## _Set up_ da análise

# In[7]:


import pandas as pd
import numpy as np


# In[8]:


black_friday = pd.read_csv("black_friday.csv")


# ## Inicie sua análise a partir daqui

# In[9]:


df = black_friday


# ## Questão 1
# 
# Quantas observações e quantas colunas há no dataset? Responda no formato de uma tuple `(n_observacoes, n_colunas)`.

# In[10]:


def q1():
    '''retorna a contagem de colunas e linhas'''
    return black_friday.shape
    # Retorne aqui o resultado da questão 1.


# ## Questão 2
# 
# Há quantas mulheres com idade entre 26 e 35 anos no dataset? Responda como um único escalar.

# In[11]:


def q2():
    '''verifica se a observação condiz com a condição e retorna a contagem'''
    df = black_friday.loc[(black_friday['Gender'] == 'F') & (black_friday['Age'] == '26-35')]
    return df.shape[0]


# ## Questão 3
# 
# Quantos usuários únicos há no dataset? Responda como um único escalar.

# In[15]:


def q3():
    '''remove duplicados e retorna a contagem de index'''
    df = black_friday.drop_duplicates(subset='User_ID')
    return df.shape[0]


# ## Questão 4
# 
# Quantos tipos de dados diferentes existem no dataset? Responda como um único escalar.

# In[8]:


def q4():
    '''retorna os tipos unicos do data frame'''
    return black_friday.dtypes.nunique()


# ## Questão 5
# 
# Qual porcentagem dos registros possui ao menos um valor null (`None`, `ǸaN` etc)? Responda como um único escalar entre 0 e 1.

# In[3]:


def q5():
    '''transforma os valores NaN/None em Booleans, some e divide pelo topa de linhas'''
    return sum(black_friday.isna().sum(axis=1) != 0) / black_friday.shape[0]


# ## Questão 6
# 
# Quantos valores null existem na variável (coluna) com o maior número de null? Responda como um único escalar.

# In[2]:


def q6():
    '''transforma os valores NaN/None em Booleans, soma e retorna a contagem'''
    return int(black_friday.isna().sum().max())


# ## Questão 7
# 
# Qual o valor mais frequente (sem contar nulls) em `Product_Category_3`? Responda como um único escalar.

# In[11]:


def q7():
    '''trata os valores null e nan e retorna o valor mais frequente'''
    black_friday.fillna(0)
    return black_friday['Product_Category_3'].value_counts().idxmax()


# ## Questão 8
# 
# Qual a nova média da variável (coluna) `Purchase` após sua normalização? Responda como um único escalar.

# In[12]:


def q8():
    '''normaliza o df usando a fórmula de normalização e retorna a média.'''
    purchase = black_friday['Purchase']
    norm = (purchase - purchase.min())/(purchase.max()-purchase.min())
    return float(norm.mean())


# ## Questão 9
# 
# Quantas ocorrências entre -1 e 1 inclusive existem da variáel `Purchase` após sua padronização? Responda como um único escalar.

# In[22]:


def q9():
    '''padroniza usando a fórmula z-score, trata os valores com a condição -1<x<1 e retorna a contagem.'''
    norm = (black_friday['Purchase'] - black_friday['Purchase'].mean()) / black_friday['Purchase'].std()
    norm = norm.loc[norm >= -1]
    norm = norm.loc[norm <= 1]
    return int(norm.shape[0])


# ## Questão 10
# 
# Podemos afirmar que se uma observação é null em `Product_Category_2` ela também o é em `Product_Category_3`? Responda com um bool (`True`, `False`).

# In[1]:


def q10():
    ''''''
    return black_friday.dropna().shape[0] == black_friday.dropna(subset = ['Product_Category_3']).shape[0]


# In[ ]:




