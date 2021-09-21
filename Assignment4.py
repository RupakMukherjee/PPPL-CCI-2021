from array import*

N=1000
u=1
dt=0.001

array_N =[]
for i in range(1,1000):
        u=(u*i+1*dt)/(i+1);
        array_N.append(u*(i+1))
   
   
print(*array_N, sep= "\n")



