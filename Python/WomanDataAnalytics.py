# -*- coding: utf-8 -*-
"""
Created on Sun Oct 21 21:38:53 2018

@author: marku
"""


#Task: overlap

a = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89]
b = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]

# Cez vbudovanu funkciu
import numpy as np
np.intersect1d(a,b)

# From scratch:

def overlap(a,b):
    temp_list = []
    for i in a:
        if( (i in b) == True):
            temp_list.append(i)
    print(list(set(temp_list)))  

overlap(a,b)        
      

# Task even numbers
a = [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

# For cyklus
def only_even(a):
    temp_list = []
    for i in a:
        if (i % 2) == 0:
            temp_list.append(i)
    return(temp_list)
    
only_even(a)   

# List comprehension syntax
[i     for i in a      if i % 2 == 0]


def is_even(x):
    return((x % 2) == 0)
    
[i     for i in a      if(is_even(i))]

# Task max 3 numbers

def max_of_3(a,b,c):
    if (a >= b) and (a >= c):
        return(a)
    elif (b >= a) and (b >= c): 
        return(b)
    else:
        return(c)
        
max_of_3(333,5,25)


## Task: birthday dictionaries

dict = {
    "Nikolas Markus": "24.2.1995", 
	 "Melissa Markus": "19.10.2000",
	 "Helena Markus": "20.5.1972"
       }


def birthday_dict(x):
    print("Welcome to the birthday dictionary. We know the birthdays of these people:")
    for names in dict:
        print(names)     

    x = input("Who's birthday do you want to look up? ")
    if(x in dict):
        print(f"{x} was born on {dict[x]}.")
    else:
        print("I have no idea")
        
birthday_dict(dict)



fullnames = ['Juraj Menko', 'Peter Novák', 'Ján Koleník', 'Martin Krajcer', 'Jozef Lipa', 'Rudolf Vrtík']

temp_list = []
for i in range(len(fullnames)):
   person = fullnames[i]
   for char in reversed(person):
       if(char != " "):
           temp_list.append(char)
       else:
           break
print("Count of all unique characters in surnames:",len(list(set(temp_list))))    
    

rev("takto)

>>> Welcome to the birthday dictionary. We know the birthdays of these people:
Albert Einstein
Benjamin Franklin
Barack Obama


>>> Who's birthday do you want to look up?
Benjamin Franklin
>>> Benjamin Franklin's birthday is 17.1.1706



# Task christmas

import datetime as dt

x = input("Enter a date: ")
dt.datetime(x)
dt.datetime()
dt.date("2018-06-01")
