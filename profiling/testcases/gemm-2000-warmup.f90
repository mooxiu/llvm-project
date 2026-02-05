program test
    implicit none

    integer(8), parameter :: N = 2
    double precision:: b
    integer(8), dimension(:, :), allocatable :: x
    integer(8), dimension(:, :), allocatable :: y
    integer(8), dimension(:, :), allocatable :: z

    allocate(x(N, N))
    allocate(y(N, N))
    allocate(z(N, N))

    x = 3
    y = 2
    z = 0

    write (*, '(A)') 'calling axpy'
    b = abs(coexecute_a(x, y, z, N))
    b = abs(coexecute_a(x, y, z, N))

    deallocate(x)
    deallocate(y)
    deallocate(z)

contains
function coexecute_a(x, y, z, n) result(sum_less)
  use omp_lib
  implicit none
  integer(8) :: n
  double precision:: sum_less
  integer(8), dimension(n, n) :: x, y, z
  double precision:: ostart, oend, allstart, allend

  write (*,*) 'n before', n
  write (*,*) 'z(1,1) before', z(1,1)
  write (*,*) 'checksum before', sum(z(1:n, 1:n))

    allstart = omp_get_wtime()
    !$omp target data map(tofrom:x,y,z)
    ostart = omp_get_wtime()
    !$omp target teams workdistribute
    z = matmul(x, y)
    !$omp end target teams workdistribute
    oend = omp_get_wtime()
    !$omp end target data
    allend = omp_get_wtime()


  write (*,*) 'n after', n
  write (*,*) 'z(1,1) after', z(1,1)
  write (*,*) 'checksum after', sum(z(1:n, 1:n))

  print *, 'Time computation: ', oend-ostart, 'seconds.'
  print *, 'Time all: ', allend-allstart, 'seconds.'

  sum_less = sum(z(1:n/2,1:n/3) - 2) / ( n * n)

end function coexecute_a
end program test
