      PROGRAM BENCH_MADLOOP
C     Per-phase-space-point timing of a MadLoop standalone output.
C
C     Input (file bench_input.txt in the current directory, i.e. the P*
C     subprocess directory, like the stock 'check' executable):
C       HEL NPTS NBLOCKS BLOCK_MS WARMUP_SWEEPS MUR
C       NPTS*NEXTERNAL lines: E PX PY PZ (MadLoop leg order)
C     HEL > 0 : ML5_0_SLOOPMATRIXHEL(P,HEL,ANS) (one helicity configuration,
C               index into MadLoop5_resources/*HelConfigs.dat)
C     HEL = 0 : ML5_0_SLOOPMATRIX(P,ANS) (helicity summed)
C     MUR > 0 : renormalization scale written to COMMON/RSCALE/ (alpha_s is
C               NOT re-run, as in the existing validation harnesses).
C
C     Output on stdout, one 'key value...' record per line (parsed by
C     run_madloop.py): results per point, per-block per-point times, and the
C     Ninja instrumentation counters (only meaningful when linked against the
C     instrumented Ninja; otherwise the weak stubs return -1).
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      INTEGER MAXPTS, MAXBLK
      PARAMETER (MAXPTS=64, MAXBLK=101)
      DOUBLE PRECISION P(0:3,NEXTERNAL,MAXPTS)
      DOUBLE PRECISION, ALLOCATABLE :: ANS(:,:)
      INTEGER ANSDIM, NPTS, HEL, NBLOCKS, NWARM, I, J, K, B
      INTEGER*8 NSWEEPS, S, T0, T1, RATE
      DOUBLE PRECISION BLOCK_MS, MUR, ONE_SWEEP, DT
      DOUBLE PRECISION CNT(8), NRANK(99), HZ, NEVAL
      DOUBLE PRECISION CHECKSUM
      DOUBLE PRECISION MU_R, ALL_MU_R
      COMMON/RSCALE/ MU_R, ALL_MU_R
      DOUBLE PRECISION NINJA_BENCH_TSC_HZ
      EXTERNAL NINJA_BENCH_TSC_HZ

      OPEN(10, FILE='bench_input.txt', STATUS='OLD')
      READ(10,*) HEL, NPTS, NBLOCKS, BLOCK_MS, NWARM, MUR
      IF (NPTS.GT.MAXPTS .OR. NBLOCKS.GT.MAXBLK) STOP 'too many'
      DO K=1,NPTS
        DO I=1,NEXTERNAL
          READ(10,*) (P(J,I,K),J=0,3)
        ENDDO
      ENDDO
      CLOSE(10)

      CALL ML5_0_GET_ANSWER_DIMENSION(ANSDIM)
      ALLOCATE(ANS(0:3,0:ANSDIM))
      CALL SETPARA('param_card.dat')
      IF (MUR.GT.0D0) MU_R = MUR

C     First evaluation of every point (includes MadLoop initialization).
      DO K=1,NPTS
        CALL EVAL(P(0,1,K), HEL, ANS)
        WRITE(*,'(A,I3,4(1X,ES24.16E3))') 'RESULT ', K,
     &    ANS(0,0), ANS(1,0), ANS(2,0), ANS(3,0)
      ENDDO

C     Warm-up sweeps (MadLoop helicity-filter check phase, caches).
      CHECKSUM = 0D0
      DO S=1,NWARM
        DO K=1,NPTS
          CALL EVAL(P(0,1,K), HEL, ANS)
          CHECKSUM = CHECKSUM + ANS(1,0)
        ENDDO
      ENDDO

C     Calibrate the number of sweeps per block.
      CALL SYSTEM_CLOCK(T0, RATE)
      DO K=1,NPTS
        CALL EVAL(P(0,1,K), HEL, ANS)
        CHECKSUM = CHECKSUM + ANS(1,0)
      ENDDO
      CALL SYSTEM_CLOCK(T1, RATE)
      ONE_SWEEP = DBLE(T1-T0)/DBLE(RATE)
      NSWEEPS = MAX(1_8, INT(BLOCK_MS*1D-3/MAX(ONE_SWEEP,1D-9),8)+1)
      WRITE(*,'(A,I12,1X,ES14.6)') 'CALIBRATION ', NSWEEPS, ONE_SWEEP

      CALL NINJA_BENCH_RESET()
      DO B=1,NBLOCKS
        CALL SYSTEM_CLOCK(T0, RATE)
        DO S=1,NSWEEPS
          DO K=1,NPTS
            CALL EVAL(P(0,1,K), HEL, ANS)
            CHECKSUM = CHECKSUM + ANS(1,0)
          ENDDO
        ENDDO
        CALL SYSTEM_CLOCK(T1, RATE)
        DT = DBLE(T1-T0)/DBLE(RATE)/DBLE(NSWEEPS*NPTS)
        WRITE(*,'(A,I4,1X,ES16.8)') 'BLOCK ', B, DT
      ENDDO
      NEVAL = DBLE(NBLOCKS)*DBLE(NSWEEPS)*DBLE(NPTS)
      CALL NINJA_BENCH_GET(CNT)
      CALL NINJA_BENCH_GET_NRANK(NRANK)
      HZ = NINJA_BENCH_TSC_HZ()
      WRITE(*,'(A,ES16.8)') 'EVALUATIONS ', NEVAL
      WRITE(*,'(A,ES16.8)') 'TSC_HZ ', HZ
      WRITE(*,'(A,8(1X,ES16.8))') 'NINJA_COUNTERS', (CNT(I),I=1,8)
      DO I=0,8
        DO J=0,10
          IF (NRANK(I*11+J+1).GT.0D0) THEN
            WRITE(*,'(A,I3,I3,1X,ES16.8)') 'NINJA_N_RANK ', I, J,
     &        NRANK(I*11+J+1)
          ENDIF
        ENDDO
      ENDDO
      WRITE(*,'(A,ES24.16E3)') 'CHECKSUM ', CHECKSUM
      END

      SUBROUTINE EVAL(P, HEL, ANS)
      IMPLICIT NONE
      INCLUDE 'nexternal.inc'
      DOUBLE PRECISION P(0:3,NEXTERNAL), ANS(0:3,0:*)
      INTEGER HEL
      IF (HEL.GT.0) THEN
        CALL ML5_0_SLOOPMATRIXHEL(P, HEL, ANS)
      ELSE
        CALL ML5_0_SLOOPMATRIX(P, ANS)
      ENDIF
      END
