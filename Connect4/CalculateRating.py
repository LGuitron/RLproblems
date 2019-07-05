import math

'''
Calculate new rating with match results (self play with old agent with known rating)
Used the formula on wikipedia where:
a = new agent
b = old agent
E = score (real score used)
'''
def calculate_rating(Rb, Ea):
    Eb = 1 - Ea
    Qb = pow(10, (Rb/400))
    Qa = (Qb/Eb) - Qb
    Ra = 400 * math.log10(Qa)
    return Ra

        
    
