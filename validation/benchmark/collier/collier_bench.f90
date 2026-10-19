! collier_bench: binary64 timing and accuracy driver for COLLIER on prepared
! rank n+2 bench lanes (validation/benchmark/collier/prepare.py).
!
! Usage:  collier_bench IN OUT MODE [BLOCKS] [MIN_BLOCK_MS] [CACHE] [REQACC] [ONLY]
!   MODE          1 = COLI, 2 = DD
!   BLOCKS        timed blocks per quantity (default 9); the median is reported
!   MIN_BLOCK_MS  minimum duration of one block in milliseconds (default 20); a negative
!                 value -R fixes the repetitions per lane and block to R (no calibration:
!                 fixed work for external instruction counting with perf stat)
!   CACHE         0 = no global cache (default), 1 = one global cache per lane
!                 (InitCacheSystem_cll, InitEvent_cll before every lane evaluation)
!   REQACC        SetReqAcc_cll (default: COLLIER's 1d-8); <= 0 keeps the default
!   ONLY          time only this quantity (e.g. total_finite); default all
!
! IN (list-directed, written by prepare.py; every value is a binary64 number):
!   nlanes nmax rmax
!   per lane:
!     npieces mu2
!     per piece:
!       N R ncoef
!       N lines: Re(m_k^2) Im(m_k^2)              k = 0..N-1
!       N-1 lines: p_k^0 p_k^1 p_k^2 p_k^3        k = 1..N-1 (p_0 = 0)
!       N(N-1)/2 lines: invariant (COLLIER MomInv order)
!       ncoef lines: n0 n1 n2 n3 Re(c) Im(c)      N(l) = sum c l0^n0 l1^n1 l2^n2 l3^n3
! A lane is the integrand of one phase-space point, decomposed into COLLIER pieces
! (mu^2 removed by pinching, each piece shifted to p_0 = 0) by
! validation/rank-n2/harness/collier_reference.py; prepare.py rounds the exact
! pieces to binary64 once, before any timing.
!
! Normalisation (validation/rank-n2/harness/collier_tensor_driver.f90): three settings of
! COLLIER's Delta parameters with mu_UV^2 = mu_IR^2 = mu2,
!   set 1: DeltaUV=0, DeltaIR1=0, DeltaIR2=pi^2/6     -> eps^0 coefficient (r_Gamma norm)
!   set 2: DeltaUV=1, DeltaIR1=1, DeltaIR2=pi^2/6     -> set1 + eps^-1 coefficient
!   set 3: DeltaUV=0, DeltaIR1=0, DeltaIR2=1+pi^2/6   -> set1 + eps^-2 coefficient
! so eps0 = set1, eps_m1 = set2 - set1, eps_m2 = set3 - set1 (eps_UV = eps_IR).
!
! OUT: one line per lane (validation pass, untimed), then timing lines:
!   LANE l errflag accflag eps0re eps0im em1re em1im em2re em2im errest absum0
!     errflag/accflag: the most negative over the three sets and all pieces; errest:
!     sum over pieces and coefficients of |c| TNerr(rank of c) (set 1, COLLIER's own
!     absolute error estimate of the tensor components); absum0: sum |c| |T| (set 1)
!   TIME quantity median_ns_per_point min_ns_per_point max_ns_per_point blocks reps
!     quantity: calls_finite      COLLIER calls of every piece, Delta set 1 only
!               contraction       contraction of stored components with the coefficients
!               total_finite      calls_finite + contraction (eps^0, the usual use)
!               total_laurent     the three Delta sets and their contractions (eps^0,
!                                 eps^-1, eps^-2: what NinjaRS returns), including the
!                                 Delta parameter changes between the sets
!               delta_switch      those parameter changes alone (SetDeltaUV_cll and
!                                 SetDeltaIR_cll for the three sets)
!               scale_switch      SetMuUV2_cll + SetMuIR2_cll (not part of the others)
!     ns per point = block time / (repetitions x lanes). A block takes every lane in turn:
!     its scale is set (untimed), then it is evaluated reps times (timed).
program collier_bench
  use COLLIER
  use, intrinsic :: iso_fortran_env, only: int64
  implicit none

  type piece_t
    integer :: N, R, ncoef
    double complex, allocatable :: mass2(:), momvec(:,:), mominv(:)
    integer, allocatable :: idx(:,:)
    double complex, allocatable :: coef(:)
    double complex, allocatable :: T(:,:,:,:,:)    ! (0:R)^4 x 3 Delta sets
    double complex, allocatable :: Tuv(:,:,:,:)
    double precision, allocatable :: Terr(:)
  end type piece_t
  type lane_t
    integer :: npieces
    double precision :: mu2
    type(piece_t), allocatable :: p(:)
  end type lane_t

  type(lane_t), allocatable :: lanes(:)
  integer :: nlanes, nmax, rmax, mode, nblocks, gcache, nargs, unit_in, unit_out, ios
  integer :: l, j, k, iset, errflag, accflag, errmin, accmin, ncache
  double precision :: min_block_ms, reqacc, pi2o6, d_uv(3), d_ir1(3), d_ir2(3)
  double precision :: re, im, errest, absum, p4(0:3)
  double complex :: acc(3), sink
  character(len=4096) :: path_in, path_out, arg, only

  pi2o6 = 4d0*atan(1d0)
  pi2o6 = pi2o6*pi2o6/6d0
  d_uv  = (/ 0d0, 1d0, 0d0 /)
  d_ir1 = (/ 0d0, 1d0, 0d0 /)
  d_ir2 = (/ pi2o6, pi2o6, 1d0 + pi2o6 /)

  nargs = command_argument_count()
  if (nargs < 3) then
    write(0,*) 'usage: collier_bench IN OUT MODE [BLOCKS] [MIN_BLOCK_MS] [CACHE] [REQACC] [ONLY]'
    stop 2
  end if
  call get_command_argument(1, path_in)
  call get_command_argument(2, path_out)
  call get_command_argument(3, arg); read(arg, *) mode
  nblocks = 9
  if (nargs >= 4) then; call get_command_argument(4, arg); read(arg, *) nblocks; end if
  min_block_ms = 20d0
  if (nargs >= 5) then; call get_command_argument(5, arg); read(arg, *) min_block_ms; end if
  gcache = 0
  if (nargs >= 6) then; call get_command_argument(6, arg); read(arg, *) gcache; end if
  reqacc = -1d0
  if (nargs >= 7) then; call get_command_argument(7, arg); read(arg, *) reqacc; end if
  only = ''
  if (nargs >= 8) call get_command_argument(8, only)

  ! ---------------------------------------------------------------- input --
  open(newunit=unit_in, file=trim(path_in), status='old', action='read', iostat=ios)
  if (ios /= 0) then
    write(0,*) 'cannot open ', trim(path_in)
    stop 2
  end if
  read(unit_in, *) nlanes, nmax, rmax
  allocate(lanes(nlanes))
  do l = 1, nlanes
    read(unit_in, *) lanes(l)%npieces, lanes(l)%mu2
    allocate(lanes(l)%p(lanes(l)%npieces))
    do j = 1, lanes(l)%npieces
      associate (pc => lanes(l)%p(j))
        read(unit_in, *) pc%N, pc%R, pc%ncoef
        allocate(pc%mass2(0:pc%N-1), pc%momvec(0:3, max(pc%N-1, 1)), &
                 pc%mominv(max(pc%N*(pc%N-1)/2, 1)), pc%idx(4, pc%ncoef), pc%coef(pc%ncoef))
        pc%momvec = (0d0, 0d0)
        pc%mominv = (0d0, 0d0)
        do k = 0, pc%N-1
          read(unit_in, *) re, im
          pc%mass2(k) = dcmplx(re, im)
        end do
        do k = 1, pc%N-1
          read(unit_in, *) p4
          pc%momvec(:, k) = dcmplx(p4, 0d0)
        end do
        do k = 1, pc%N*(pc%N-1)/2
          read(unit_in, *) re
          pc%mominv(k) = dcmplx(re, 0d0)
        end do
        do k = 1, pc%ncoef
          read(unit_in, *) pc%idx(:, k), re, im
          pc%coef(k) = dcmplx(re, im)
        end do
        allocate(pc%T(0:pc%R, 0:pc%R, 0:pc%R, 0:pc%R, 3), pc%Tuv(0:pc%R, 0:pc%R, 0:pc%R, 0:pc%R), &
                 pc%Terr(0:pc%R))
        pc%T = (0d0, 0d0)
        pc%Tuv = (0d0, 0d0)
        pc%Terr = 0d0
      end associate
    end do
  end do
  close(unit_in)

  call Init_cll(max(nmax, 2), max(rmax, 1), '')
  call SetMode_cll(mode)
  if (reqacc > 0d0) call SetReqAcc_cll(reqacc)
  if (gcache /= 0) then
    ncache = nlanes
    call InitCacheSystem_cll(ncache, max(nmax, 2))
  end if

  open(newunit=unit_out, file=trim(path_out), status='replace', action='write')

  ! ------------------------------------------- validation pass (untimed) --
  do l = 1, nlanes
    call set_scale(l)
    errmin = 0
    accmin = 0
    do iset = 1, 3
      call SetDeltaUV_cll(d_uv(iset))
      call SetDeltaIR_cll(d_ir1(iset), d_ir2(iset))
      call InitErrFlag_cll()
      call InitAccFlag_cll()
      call lane_calls(l, iset)
      call GetErrFlag_cll(errflag)
      call GetAccFlag_cll(accflag)
      errmin = min(errmin, errflag)
      accmin = min(accmin, accflag)
    end do
    acc = (0d0, 0d0)
    errest = 0d0
    absum = 0d0
    do j = 1, lanes(l)%npieces
      associate (pc => lanes(l)%p(j))
        do iset = 1, 3
          acc(iset) = acc(iset) + contract(pc, iset)
        end do
        do k = 1, pc%ncoef
          errest = errest + abs(pc%coef(k))*pc%Terr(sum(pc%idx(:, k)))
          absum = absum + abs(pc%coef(k))*abs(pc%T(pc%idx(1,k), pc%idx(2,k), pc%idx(3,k), pc%idx(4,k), 1))
        end do
      end associate
    end do
    write(unit_out, '(A,1X,I0,1X,I0,1X,I0,8(1X,ES25.16E3))') 'LANE', l - 1, errmin, accmin, &
      acc(1), acc(2) - acc(1), acc(3) - acc(1), errest, absum
  end do
  flush(unit_out)

  ! ------------------------------------------------------------- timing --
  sink = (0d0, 0d0)
  call SetDeltaUV_cll(d_uv(1))
  call SetDeltaIR_cll(d_ir1(1), d_ir2(1))
  call timed('calls_finite', 1)
  call timed('contraction', 2)
  call timed('total_finite', 3)
  call timed('total_laurent', 4)
  call timed('delta_switch', 5)
  call timed('scale_switch', 6)
  write(unit_out, '(A,1X,2(ES25.16E3,1X))') 'SINK', sink
  close(unit_out)

contains

  ! The renormalisation scale of lane l (mu_UV^2 = mu_IR^2). Every call costs two
  ! DDsetparam calls, which determine the machine precision in a loop (a few
  ! microseconds), so the timed quantities set it outside the timed region, once per
  ! lane and block, as an application does with a fixed scale; 'scale_switch'
  ! measures it.
  subroutine set_scale(l)
    integer, intent(in) :: l
    call SetMuUV2_cll(lanes(l)%mu2)
    call SetMuIR2_cll(lanes(l)%mu2)
  end subroutine set_scale

  ! All COLLIER calls of lane l, Delta set iset (already set), into p%T(...,iset).
  subroutine lane_calls(l, iset)
    integer, intent(in) :: l, iset
    integer :: j
    if (gcache /= 0) call InitEvent_cll(l)
    do j = 1, lanes(l)%npieces
      associate (pc => lanes(l)%p(j))
        if (pc%N == 1) then
          call Aten_cll(pc%T(:,:,:,:,iset), pc%Tuv, pc%mass2(0:0), pc%R, pc%Terr)
        else
          call TNten_cll(pc%T(:,:,:,:,iset), pc%Tuv, pc%momvec(:, 1:pc%N-1), &
                         pc%mominv(1:pc%N*(pc%N-1)/2), pc%mass2(0:pc%N-1), pc%N, pc%R, pc%Terr)
        end if
      end associate
    end do
  end subroutine lane_calls

  double complex function contract(pc, iset)
    type(piece_t), intent(in) :: pc
    integer, intent(in) :: iset
    integer :: k
    contract = (0d0, 0d0)
    do k = 1, pc%ncoef
      contract = contract + pc%coef(k)*pc%T(pc%idx(1,k), pc%idx(2,k), pc%idx(3,k), pc%idx(4,k), iset)
    end do
  end function contract

  ! One evaluation of quantity q for lane l (scale already set); accumulates sink.
  subroutine lane_eval(l, q)
    integer, intent(in) :: l, q
    integer :: j, iset
    double complex :: s
    select case (q)
    case (1)
      call lane_calls(l, 1)
    case (2)
      s = (0d0, 0d0)
      do j = 1, lanes(l)%npieces
        s = s + contract(lanes(l)%p(j), 1)
      end do
      sink = sink + s
    case (3)
      call lane_calls(l, 1)
      s = (0d0, 0d0)
      do j = 1, lanes(l)%npieces
        s = s + contract(lanes(l)%p(j), 1)
      end do
      sink = sink + s
    case (4)
      do iset = 1, 3
        call SetDeltaUV_cll(d_uv(iset))
        call SetDeltaIR_cll(d_ir1(iset), d_ir2(iset))
        call lane_calls(l, iset)
      end do
      s = (0d0, 0d0)
      do j = 1, lanes(l)%npieces
        s = s + contract(lanes(l)%p(j), 1) + contract(lanes(l)%p(j), 2) + contract(lanes(l)%p(j), 3)
      end do
      sink = sink + s
    case (5)
      ! the parameter changes of total_laurent alone
      do iset = 1, 3
        call SetDeltaUV_cll(d_uv(iset))
        call SetDeltaIR_cll(d_ir1(iset), d_ir2(iset))
      end do
    case (6)
      call set_scale(l)
    end select
  end subroutine lane_eval

  double precision function now_ns()
    integer(int64) :: count, rate
    call system_clock(count, rate)
    now_ns = dble(count)*(1d9/dble(rate))
  end function now_ns

  ! One block of quantity q: every lane in turn, its scale set untimed, then reps timed
  ! evaluations; returns the summed timed nanoseconds.
  double precision function block_ns(q, reps)
    integer, intent(in) :: q, reps
    integer :: l, r
    double precision :: t0
    block_ns = 0d0
    do l = 1, nlanes
      call set_scale(l)
      t0 = now_ns()
      do r = 1, reps
        call lane_eval(l, q)
      end do
      block_ns = block_ns + (now_ns() - t0)
    end do
    if (q == 4 .or. q == 5) then
      call SetDeltaUV_cll(d_uv(1))
      call SetDeltaIR_cll(d_ir1(1), d_ir2(1))
    end if
  end function block_ns

  subroutine timed(name, q)
    character(len=*), intent(in) :: name
    integer, intent(in) :: q
    integer :: reps, b
    double precision :: t, block(nblocks), tmp
    integer :: i1, i2
    if (len_trim(only) > 0 .and. trim(only) /= name) return
    ! warm-up and calibration: double the repetitions until a block lasts min_block_ms
    t = block_ns(q, 1)
    if (min_block_ms < 0d0) then
      reps = max(1, nint(-min_block_ms))
    else
      reps = 1
      do
        t = block_ns(q, reps)
        if (t >= min_block_ms*1d6 .or. reps >= 2**26) exit
        reps = reps*2
      end do
    end if
    do b = 1, nblocks
      block(b) = block_ns(q, reps)/dble(reps)/dble(nlanes)
    end do
    ! insertion sort for the median
    do i1 = 2, nblocks
      tmp = block(i1)
      i2 = i1 - 1
      do while (i2 >= 1)
        if (block(i2) <= tmp) exit
        block(i2 + 1) = block(i2)
        i2 = i2 - 1
      end do
      block(i2 + 1) = tmp
    end do
    write(unit_out, '(A,1X,A,3(1X,ES16.8E3),1X,I0,1X,I0)') 'TIME', name, &
      block((nblocks + 1)/2), block(1), block(nblocks), nblocks, reps
    flush(unit_out)
  end subroutine timed

end program collier_bench
