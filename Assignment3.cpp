#include <iostream>
using namespace std;

int main()
{
    
    
    double x=0, v=1, dt=0.001,N=1000,i;
    
    for (double i=1; i<=N;i++) {
        x=x+v*dt;
    }
    

       cout <<x<<endl;


return 0;
}
