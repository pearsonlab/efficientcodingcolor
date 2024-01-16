# -*- coding: utf-8 -*-
"""
Created on Thu Jan  4 10:48:59 2024

@author: Rache
"""

from sympy import *
init_printing()

b1,b2,b3,b4,b5,b6,b7,b8,b9 = symbols("b1 b2 b3 b4 b5 b6 b7 b8 b9")
u = symbols("u")

B = Matrix([[b1, b2, b3], [b4, b5, b6], [b7, b8, b9]])

solve([B*B.T - eye(3), b1 - u/2, b2 - u/3, b3 - u/4], [b1,b2,b3,b4,b5,b6,b7,b8,b9])

## Attempt number 2!!!

a, b, y = symbols("a b y")
p1,p2,p3 = symbols("p1 p2 p3")
b1 = cos(b)*cos(y)
b2 = sin(a)*sin(b)*sin(y) + cos(a)*cos(y)
b3 = cos(a)*cos(b)

ans = solve([b1 - sqrt(u/p1), b2 - sqrt(u/p2), b3 - sqrt(u/p3)], [a,b,y,u])