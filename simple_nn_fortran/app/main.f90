program main
  use nn_utils, only: nn_tran
  use stdlib_math, only: linspace
  use stdlib_kinds, only: int32, dp
  implicit none

  real(dp), allocatable :: x(:), y(:), param(:)
  integer(int32)        :: data_size, param_size, tran_loop, i
  real(dp)              :: PI, learing_rate

  PI = acos(-1.0_dp)
  data_size = 2000
  param_size = 4

  allocate(x(data_size))
  allocate(y(data_size))
  allocate(param(param_size))

  call random_number(param)

  x = linspace(-PI, PI, data_size)
  do i = 1, data_size
    y(i) = sin(x(i))
  end do

  ! use y = a + b * x + c * x**2 + d * x**3 to fit sin(x)
  learing_rate = 1e-06
  tran_loop = 2000
  call nn_tran(x, y, param, learing_rate, tran_loop)

  print *, 'Result: y =', param(1), '+', param(2), '* x +', param(3), '* x^2 +', param(4), '* x^3'

end program main
