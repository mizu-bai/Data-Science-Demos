module nn_utils
  use stdlib_kinds, only: int32, dp
  implicit none
  private

  public :: nn_tran
  
contains

  subroutine nn_tran(x, y, param, learning_rate, tran_loop)
    ! parameters
    real(dp), intent(in)       :: x(:), y(:), learning_rate
    integer(int32), intent(in) :: tran_loop
    real(dp), intent(inout)    :: param(:)

    real(dp), allocatable      :: y_pred(:), grad_y_pred(:), grad_param(:)
    real(dp)                   :: loss
    integer(int32)             :: t, i, data_size, param_size

    ! allocate
    data_size  = size(y)
    param_size = size(param)

    allocate(y_pred(data_size))
    allocate(grad_y_pred(data_size))
    allocate(grad_param(param_size))

    ! トレーニングを始める！
    do t = 1, tran_loop
      ! initialize
      y_pred(:)      = 0._dp
      grad_param(:)  = 0._dp
      grad_y_pred(:) = 0._dp
      loss           = 0._dp

      ! forward pass
      do i = 1, data_size
        y_pred(i) = param(1) + param(2) * x(i) + param(3) * x(i)**2 + param(4) * x(i)**3 
      end do

      ! compute loss
      do i = 1, data_size
        loss = loss + (y_pred(i) - y(i))**2
      end do

      if (mod(t, 100) == 0) then
        print *, 't = ', t, 'loss = ', loss
      end if

      ! backprop
      do i = 1, data_size
        grad_y_pred(i) = 2.0 * (y_pred(i) - y(i))
      end do

      do i = 1, data_size
        grad_param(1) = grad_param(1) + grad_y_pred(i)
        grad_param(2) = grad_param(2) + grad_y_pred(i) * x(i)
        grad_param(3) = grad_param(3) + grad_y_pred(i) * x(i)**2
        grad_param(4) = grad_param(4) + grad_y_pred(i) * x(i)**3
      end do

      ! update param
      do i = 1, param_size
        param(i) = param(i) - learning_rate * grad_param(i)
      end do

    end do

    ! deallocate
    deallocate(y_pred)
    deallocate(grad_param)
    deallocate(grad_y_pred)

  end subroutine nn_tran
  
end module nn_utils
