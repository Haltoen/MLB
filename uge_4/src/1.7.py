import math
def vc_bound (n):
    return math.sqrt(8 * math.log((2*((2*n)**11))/0.01)/n)


precision = 0.01
n = 15000000 # initial n, set close to terminaltion point

while True:
    bound = vc_bound(n)
    if bound <= precision:
        break
    else: 
        n += 1
    
print("first valid n:",n)
