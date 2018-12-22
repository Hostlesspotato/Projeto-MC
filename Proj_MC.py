# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 19:16:36 2018

@author: smart jr
"""


import numpy  as np
import matplotlib.pyplot as plt
from math import * 

G = 17

S = 325225
L = 3.25
R = 32.2
C = 0.352e-3


def charge_q(t):
     
    
     
     prod_1 = -R/(2*L)
     expr_1 = G * np.exp(t * prod_1)
     
     l_c = L*C
     r_l = np.power(R/(2*L), 2)
     prod_2 = t* np.sqrt(1/l_c - r_l)
     expr_2 = np.cos(prod_2)
     
     return expr_1 * expr_2
      # return G*(np.exp(-(R/(2*L)) * t))*np.cos(t * np.sqrt((1/L*C) - np.power(R,2)/(4*np.power(L,2))))


def  derivate_of_function(function, value):
    h = 0.00000000001
    
    top = function(value + h ) - function(value)
    bottom = h
    slope = top / bottom
    
    return float ("%.3f" % slope)

def current_i(t):
    return derivate_of_function(charge_q,t)
    
t1 = np.arange(0.0, 1.1, 0.005)


plt.plot(t1, charge_q(t1), 'k')
plt.show()




r = 1* (-1)

def Euler_method(list_of_values,t, h):
    """ The interval of t is from [0,1], the first time the T_n is calculated it will be  """
    previous_t = T_n(t,h)
    
    if previous_t - h  == 0 :
        list_of_values.append((r*R*G)/(2*L))
        return (r*R*G)/(2*L)
    else:    
        """ get the index of the last result computed by the Euler Method """
        index = len(list_of_values) - 1
        """get the last result computed by the Euler Method """
        result = list_of_values[index] + (h*F_t(previous_t))
        """ add  the new value computed by  the method to the list of the computed values"""
        list_of_values.append(result)
        
        return result

def Euler_method_1(list_of_values,min_val,max_val, h):
    """ The interval of t is from [0,1], the first time the T_n is calculated it will be  """  
   
    while(min_val <= max_val) : 
        if min_val == 0 :
            list_of_values.append((r*R*G)/(2*L))
        else :
            """ get the index of the last result computed by the Euler Method """
            index = len(list_of_values) - 1
            """get the last result computed by the Euler Method """
            result = list_of_values[index] + (h*F_t(T_n(min_val,h)))
            """ add  the new value computed by  the method to the list of the computed values"""
            list_of_values.append(result)
        
        return Euler_method_1(list_of_values,T_n(min_val,h),max_val,h)

"""  Valor inicial que e dado """
last_y =[(-R*G)/(2*L)]
""" To """
last_t = [0]



def grafico_a(a,b,function_param):
    xtrace=[]
    ytrace=[]
    for t in np.arange(a,b,0.005):
        xtrace.append(t)
        ytrace.append(function_param(t))
#    print(xtrace,ytrace)
    plt.xlabel('Time')
    plt.ylabel('q(t)')
    plt.axis([0, max(xtrace), min(ytrace), max(ytrace)])
    plt.plot(xtrace,ytrace,'-')

def R_t(x):
    return R*x + G*C*np.sin(x)


""" the F(t) function in the Euler Method  """
def F_t(t):
    
    return (-R_t(last_y[0]) - charge_q(t)/C) / L
def Euler(t,h):
    if t == 0:
        return last_y[0]
    
    else:
        computed_y = last_y[0] + h*F_t(t)
        last_y[0] = computed_y
        
        return computed_y

def grafico_Euler(a,b,h):
    xtrace=[]
    ytrace=[]
    for t in np.arange(a,b + 0.1,h):
        xtrace.append(t)
        ytrace.append(Euler(t,h))

    plt.xlabel('Time')
    plt.ylabel('Y(t)')
    plt.axis([0, max(xtrace), min(ytrace), max(ytrace)])
    plt.plot(xtrace,ytrace,'-')



    






            
    