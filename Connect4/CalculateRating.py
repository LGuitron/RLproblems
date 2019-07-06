import math
        
'''
Calculate Score from a 2 player match
'''
def calculate_rating_two_games(Rb, Ea):
    Ra = Rb + (Ea - 0.5)*2
    return Ra
