x=0
v=1
dt=0.001
N=1000

for i in range(1):
    for N in range (1,1000):
        x=x+v*dt
    
print ("%.2f"%round(x,2))

