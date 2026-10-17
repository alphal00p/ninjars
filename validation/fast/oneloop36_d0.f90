! Evaluate D0 rows (22 reals per line: 11 complex Ninja box arguments ending
! with mu^2) with a OneLOop module, e.g. MG5_aMC_v3_7_0 OneLOop-3.6:
!   gfortran -O2 -I$OLO oneloop36_d0.f90 $OLO/libavh_olo.a -o d0 && ./d0 < rows.txt
program d0check
  use avh_olo
  implicit none
  integer, parameter :: dp = kind(1d0)
  real(dp) :: v(22)
  complex(dp) :: a(11), rslt(0:2)
  integer :: ios
  call olo_unit(-1, 'all')
  do
    read(*, *, iostat=ios) v
    if (ios /= 0) exit
    a = cmplx(v(1:21:2), v(2:22:2), kind=dp)
    call olo_d0(rslt, a(1), a(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9), a(10), sqrt(real(a(11))))
    write(*, '(2es26.17)') rslt(0)
  end do
end program
