#coding=utf8
"""
Created on Tue Jan 19 23:55:41 2024

@author: Neal LONG
"""

class Person:
    
    def __init__(self, birth_year = 2000, name = 'Neal'):
        self.birth_year = birth_year
        self.name = name
        self.age =  2023 - self.birth_year
            
    def update_age(self, year):
        self.age =  year - self.birth_year
        
    def print_age(self):
        print("Age of", self.name, "is", self.age)
        
if __name__ == "__main__":
    sam = Person(1998,"Sam")
    sam.print_age()
    neal = Person()
    neal.print_age()
    neal.update_age(2024)
    print("===After updating Neal===")
    neal.print_age()
    sam.print_age()
    
    a = Person
    b = Person()
    print(isinstance(a, Person), isinstance(b, Person))

