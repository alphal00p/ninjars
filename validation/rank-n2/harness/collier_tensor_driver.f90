! collier_tensor_driver: tensor components of one-loop integrals via COLLIER.
!
! Usage:  collier_tensor_driver JOBS.in OUT.txt [MODE] [REQACC]
!   MODE   1 = COLI (default), 2 = DD, 3 = both (COLLIER returns the better one)
!   REQACC COLLIER target precision (default COLLIER's 1d-8)
!
! JOBS.in (list-directed):
!   njobs nmax rmax
!   per job:
!     N R
!     mu2                        (mu_UV^2 = mu_IR^2, the dim-reg scale)
!     N lines: Re(m_k^2) Im(m_k^2)            k = 0..N-1
!     N-1 lines: p_k^0 p_k^1 p_k^2 p_k^3      k = 1..N-1 (p_0 = 0)
!     N(N-1)/2 lines: invariant (COLLIER MomInv order)
!
! COLLIER integral (manual eq. tensorint):
!   T^{N,mu1..muP} = (2 pi mu)^(4-D)/(i pi^2) int d^Dq q^mu1..q^muP / prod_k [(q+p_k)^2 - m_k^2]
! Output, per job, every component TNten(n0,n1,n2,n3), n0+..+n3 <= R (contravariant
! indices, n_mu = number of indices equal to mu) for three settings of COLLIER's
! Delta parameters:
!   set 1: DeltaUV=0, DeltaIR1=0, DeltaIR2=pi^2/6   -> finite part, Ellis-Zanderighi/OneLOop norm
!   set 2: DeltaUV=1, DeltaIR1=1, DeltaIR2=pi^2/6   -> set1 + (1/eps coefficient, eps_UV = eps_IR)
!   set 3: DeltaUV=0, DeltaIR1=0, DeltaIR2=1+pi^2/6 -> set1 + (1/eps^2 coefficient)
! (manual sec. 2: c(eps)/(r_Gamma (4pi)^eps) = 1 + eps^2 pi^2/6, i.e. DeltaIR2 -> DeltaIR2 + pi^2/6
! switches to the r_Gamma normalisation used by QCDLoop/OneLOop/Ninja).
! Lines written:
!   JOB ijob N R errflag(min over sets) accflag(max over sets)
!   ERR r TNerr(r)   (set 1)                    r = 0..R
!   n0 n1 n2 n3  Re(fin) Im(fin)  Re(m1) Im(m1)  Re(m2) Im(m2)  Re(uv) Im(uv)
! where fin = set1, m1 = set2-set1, m2 = set3-set1, uv = TNuv of set 1 (COLLIER's UV-pole
! coefficient; diagnostic only: in COLI mode TBuv(0,0,0,0) of N=2 came back 0, so the
! harness takes the 1/eps coefficient from the Delta difference m1, never from uv).
program collier_tensor_driver
  use COLLIER
  implicit none
  integer :: njobs, nmax, rmax, ijob, N, R, k, iset, n0, n1, n2, n3, errflag, accflag
  integer :: errmin, accmax, mode, nargs, ios, unit_in, unit_out
  double precision :: mu2, re, im, reqacc, p(0:3), pi2o6
  double precision :: d_uv(3), d_ir1(3), d_ir2(3)
  character(len=4096) :: path_in, path_out, arg
  double complex, allocatable :: mass2(:), MomInv(:), MomVec(:,:)
  double complex, allocatable :: TN(:,:,:,:), TNuv(:,:,:,:), TNs(:,:,:,:,:), TNuv1(:,:,:,:)
  double precision, allocatable :: TNerr(:), TNerr1(:)

  pi2o6 = 4d0*atan(1d0)
  pi2o6 = pi2o6*pi2o6/6d0
  d_uv  = (/ 0d0, 1d0, 0d0 /)
  d_ir1 = (/ 0d0, 1d0, 0d0 /)
  d_ir2 = (/ pi2o6, pi2o6, 1d0 + pi2o6 /)

  nargs = command_argument_count()
  if (nargs < 2) then
    write(0,*) 'usage: collier_tensor_driver JOBS.in OUT.txt [MODE] [REQACC]'
    stop 2
  end if
  call get_command_argument(1, path_in)
  call get_command_argument(2, path_out)
  mode = 1
  if (nargs >= 3) then
    call get_command_argument(3, arg)
    read(arg, *) mode
  end if
  reqacc = -1d0
  if (nargs >= 4) then
    call get_command_argument(4, arg)
    read(arg, *) reqacc
  end if

  open(newunit=unit_in, file=trim(path_in), status='old', action='read', iostat=ios)
  if (ios /= 0) then
    write(0,*) 'cannot open ', trim(path_in)
    stop 2
  end if
  read(unit_in, *) njobs, nmax, rmax

  call Init_cll(max(nmax, 2), max(rmax, 1), '')
  call SetMode_cll(mode)
  if (reqacc > 0d0) call SetReqAcc_cll(reqacc)

  open(newunit=unit_out, file=trim(path_out), status='replace', action='write')

  do ijob = 1, njobs
    read(unit_in, *) N, R
    read(unit_in, *) mu2
    allocate(mass2(0:max(N-1,1)), MomVec(0:3, max(N-1,1)), MomInv(max(N*(N-1)/2,1)))
    mass2 = (0d0, 0d0)
    MomVec = (0d0, 0d0)
    MomInv = (0d0, 0d0)
    do k = 0, N-1
      read(unit_in, *) re, im
      mass2(k) = dcmplx(re, im)
    end do
    do k = 1, N-1
      read(unit_in, *) p
      MomVec(:, k) = dcmplx(p, 0d0)
    end do
    do k = 1, N*(N-1)/2
      read(unit_in, *) re
      MomInv(k) = dcmplx(re, 0d0)
    end do

    allocate(TN(0:R,0:R,0:R,0:R), TNuv(0:R,0:R,0:R,0:R), TNs(0:R,0:R,0:R,0:R,3), &
             TNuv1(0:R,0:R,0:R,0:R), TNerr(0:R), TNerr1(0:R))
    call SetMuUV2_cll(mu2)
    call SetMuIR2_cll(mu2)
    errmin = 0
    accmax = 0
    do iset = 1, 3
      call SetDeltaUV_cll(d_uv(iset))
      call SetDeltaIR_cll(d_ir1(iset), d_ir2(iset))
      call InitErrFlag_cll()
      call InitAccFlag_cll()
      TN = (0d0, 0d0)
      TNuv = (0d0, 0d0)
      if (N == 1) then
        call Aten_cll(TN, TNuv, mass2(0:0), R, TNerr)
      else
        call TNten_cll(TN, TNuv, MomVec(:, 1:N-1), MomInv(1:N*(N-1)/2), mass2(0:N-1), N, R, TNerr)
      end if
      call GetErrFlag_cll(errflag)
      call GetAccFlag_cll(accflag)
      errmin = min(errmin, errflag)
      accmax = max(accmax, accflag)
      TNs(:,:,:,:,iset) = TN
      if (iset == 1) then
        TNuv1 = TNuv
        TNerr1 = TNerr
      end if
    end do

    write(unit_out, '(A,1X,I0,1X,I0,1X,I0,1X,I0,1X,I0)') 'JOB', ijob, N, R, errmin, accmax
    do k = 0, R
      write(unit_out, '(A,1X,I0,1X,ES25.16E3)') 'ERR', k, TNerr1(k)
    end do
    do n0 = 0, R
      do n1 = 0, R - n0
        do n2 = 0, R - n0 - n1
          do n3 = 0, R - n0 - n1 - n2
            write(unit_out, '(4(I0,1X),8(ES25.16E3,1X))') n0, n1, n2, n3, &
              TNs(n0,n1,n2,n3,1), TNs(n0,n1,n2,n3,2) - TNs(n0,n1,n2,n3,1), &
              TNs(n0,n1,n2,n3,3) - TNs(n0,n1,n2,n3,1), TNuv1(n0,n1,n2,n3)
          end do
        end do
      end do
    end do
    deallocate(mass2, MomVec, MomInv, TN, TNuv, TNs, TNuv1, TNerr, TNerr1)
  end do
  close(unit_out)
  close(unit_in)
end program collier_tensor_driver
