program Assignment3

integer :: N

x=0
v=1
dt=0.001
 N=1000

do i=1,N
  x=x+v*dt
end do
  print*,x

end program Assignment3
