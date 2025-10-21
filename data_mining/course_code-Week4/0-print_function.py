# -*- coding: utf-8 -*-
"""
Created on Sat Feb 25 16:42:32 2023

@author: Neal
"""

# Defining a class
class Test: 
    def __init__(self, a, b): 
        self.a = a 
        self.b = b 
      
    def __repr__(self): 
        return "REPR a:{} b: {}" .format(self.a, self.b) 
    
    def __str__(self): 
        return "Joke a:{} b: {}" .format(4536, "hahah") 
  
t = Test(1234, 5678) 
  
# This calls __str__() 
print(t) 
  
# # # This calls __repr__() 
print([t])

a = 'MDS5020'
print(a) 
print([a]) 