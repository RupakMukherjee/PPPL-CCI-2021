#include <iostream>

using namespace std;

int main()
{
    int N=1000, i;
    double u=1, dt=0.001;
    
    double array[N];
    
    for(int i =1; i<N; i++){
        u=(u*i+1*dt)/(i+1);
        cout<<u*(i+1)<<endl;
        
    }
    return 0;
}
