
program test
    implicit none

    integer, parameter :: N = 20000
    integer :: i
    double precision :: b
    double precision, dimension(:, :), allocatable :: x
    double precision, dimension(:, :), allocatable :: y

    allocate(x(N, N))
    allocate(y(N, N))

    do i = 1, N
      x(:, i) = i
    end do
    y = 0

    write (*, '(A)') 'calling axpy'
    b = abs(coexecute_a(x, y, N))
    b = abs(coexecute_a(x, y, N))

    deallocate(x)
    deallocate(y)

contains
function coexecute_a(x, y, n) result(sum_less)
  use omp_lib
  implicit none
  integer :: n
  double precision :: sum_less
  double precision, dimension(n, n) :: x, y
  double precision :: ostart, oend, allstart, allend

  write (*,*) 'n before', n
  write (*,*) 'x(1,2) before', x(1,2)  
  write (*,*) 'y(1,2) before', y(1,2)  ! should be 0
  write (*,*) 'checksum before', sum(y(1:n, 1:n))

    allstart = omp_get_wtime()
    !$omp target data map(tofrom:x,y)
    ostart = omp_get_wtime()
    !$omp target teams workdistribute
    y = transpose(x)
    !$omp end target teams workdistribute
    oend = omp_get_wtime()
    !$omp end target data
    allend = omp_get_wtime()


  write (*,*) 'n after', n
  write (*,*) 'x(1,2) after', x(1,2)  
  write (*,*) 'y(1,2) after', y(1,2)
  write (*,*) 'checksum after', sum(y(1:n, 1:n))

  print *, 'Time computation: ', oend-ostart, 'seconds.'
  print *, 'Time all: ', allend-allstart, 'seconds.'

  sum_less = sum(y(1:n/2,1:n/3) - 2) / ( n * n)

end function coexecute_a
end program test
