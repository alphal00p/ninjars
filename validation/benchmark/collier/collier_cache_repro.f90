! collier_cache_repro: COLLIER's global cache returns wrong 7-point rank-9 tensor integrals.
!
! One cache, one TNten_cll call per event, the same kinematic point in every event and no
! parameter change: the usage of the COLLIER manual (section "Using the cache system":
! InitCacheSystem_cll(ncache, Nmax), then InitEvent_cll(cacheNr) before the integral calls
! of every phase-space point, the calls in the same order). The tensor of every event is
! compared with the same call made before the cache was switched on.
!
!   gfortran -O2 -I<collier>/modules collier_cache_repro.f90 <collier>/libcollier.a
!   ./a.out [R] [CACHE_NMAX] [EVENTS]      defaults: 9 7 4
!
! COLLIER 1.2.10, COLI branch, binary64 (validation/benchmark/collier/build.sh, precise and
! stock builds give the same picture). Output per event: the event number, COLLIER's
! accuracy flag, and max |T_cache - T_nocache| / max |T_nocache| over the tensor components.
!   R = 9, cache Nmax 7 (default):  events 1, 2: 0;  every event >= 3: 3.9, accuracy flag -2
!   R = 9, cache Nmax 5 or 6:       the same as Nmax 7
!   R = 9, cache Nmax 3 or 4:       events >= 3: 8.0e-6, accuracy flag 0
!   R = 9, cache Nmax 2:            0 (bitwise identical)
!   R = 8, cache Nmax 7:            0 (bitwise identical)
! Events 1 and 2 are COLLIER's two initialisation events of a cache (no reuse); from event 3
! on the cache reuses the integrals it identified as identical, and the error stays the same
! in its fully optimised mode (event 14 on; ./a.out 9 7 20). The kinematics are the first
! phase-space point of validation/rank-n2/cases/rn2-int-mixed-n7-r9.json (integer momenta,
! real, complex and zero masses); every 7-point rank-9 lane of the benchmark behaves alike.
program collier_cache_repro
  use COLLIER
  implicit none
  integer, parameter :: N = 7
  integer :: R, cache_nmax, nevents, ev, i, j, k, line, cnt, accflag
  double precision :: p(0:3, 0:N-1), d(0:3)
  double complex :: momvec(0:3, N-1), mominv(N*(N-1)/2), mass2(0:N-1)
  double complex, allocatable :: T(:,:,:,:), Tref(:,:,:,:), Tuv(:,:,:,:)
  double precision, allocatable :: Terr(:)
  character(len=32) :: arg

  R = 9
  cache_nmax = N
  nevents = 4
  if (command_argument_count() >= 1) then; call get_command_argument(1, arg); read(arg, *) R; end if
  if (command_argument_count() >= 2) then; call get_command_argument(2, arg); read(arg, *) cache_nmax; end if
  if (command_argument_count() >= 3) then; call get_command_argument(3, arg); read(arg, *) nevents; end if

  ! propagator momenta p_k (p_0 = 0) and squared masses m_k^2
  p(:, 0) = (/    0d0,   0d0,   0d0,   0d0 /)
  p(:, 1) = (/  309d0,  75d0,  30d0, -44d0 /)
  p(:, 2) = (/  123d0,  58d0,  72d0,  -3d0 /)
  p(:, 3) = (/  -44d0, -40d0,  93d0, -54d0 /)
  p(:, 4) = (/ -154d0,  12d0, 139d0, -64d0 /)
  p(:, 5) = (/  347d0, 155d0,  -4d0, -77d0 /)
  p(:, 6) = (/  131d0, 100d0, -33d0,   2d0 /)
  mass2 = (/ dcmplx(8315.178393760001d0, 0d0), dcmplx(8315.178393760001d0, 0d0), &
             dcmplx(6464.160000000001d0, -168.84000000000003d0), &
             dcmplx(6464.160000000001d0, -168.84000000000003d0), &
             dcmplx(100d0, 0d0), dcmplx(29929d0, -259.5d0), dcmplx(0d0, 0d0) /)
  do k = 1, N - 1
    momvec(:, k) = dcmplx(p(:, k), 0d0)
  end do
  ! MomInv in the manual's order: (p_i - p_j)^2 with i = j + line (mod N)
  cnt = 0
  do line = 1, N/2
    do j = 0, N - 1
      i = mod(j + line, N)
      d = p(:, i) - p(:, j)
      cnt = cnt + 1
      mominv(cnt) = dcmplx(d(0)**2 - d(1)**2 - d(2)**2 - d(3)**2, 0d0)
    end do
  end do

  call Init_cll(N, R, '')
  call SetMode_cll(1)
  allocate(T(0:R,0:R,0:R,0:R), Tref(0:R,0:R,0:R,0:R), Tuv(0:R,0:R,0:R,0:R), Terr(0:R))

  ! reference: the global cache is off after Init_cll
  Tref = (0d0, 0d0)
  call TNten_cll(Tref, Tuv, momvec, mominv, mass2, N, R, Terr)

  call InitCacheSystem_cll(1, cache_nmax)
  write(*, '(A,I0,A,I0)') '# N = 7, R = ', R, ', cache Nmax = ', cache_nmax
  write(*, '(A)') '# event accflag max|T_cache - T_nocache|/max|T_nocache|'
  do ev = 1, nevents
    call InitEvent_cll(1)
    T = (0d0, 0d0)
    call TNten_cll(T, Tuv, momvec, mominv, mass2, N, R, Terr)
    call GetAccFlag_cll(accflag)
    write(*, '(I5,1X,I3,1X,ES10.3)') ev, accflag, maxval(abs(T - Tref))/maxval(abs(Tref))
  end do
end program collier_cache_repro
