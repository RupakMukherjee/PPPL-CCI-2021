program Assignment4


real, dimension(1000) :: u
integer :: i, N

u=1
dx=0.001
N=1000

do i=1,N
u(i+1)=u(i)+1*dx

    print*,u(i+1)

end do
end program Assignment4
