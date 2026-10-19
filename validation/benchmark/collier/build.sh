#!/usr/bin/env bash
# Build COLLIER 1.2.10 in binary64 and the timing driver collier_bench.f90.
#
#   validation/benchmark/collier/build.sh [BUILD_DIR]
#
# BUILD_DIR defaults to $COLLIER_BENCH_BUILD or <repo>/target/benchmark/collier-1.2.10
# (gitignored). The tarball is downloaded from hepforge into BUILD_DIR/download (a new,
# empty directory), checked against its sha256 and unpacked; the source is not modified.
# Two binary64 variants are built from copies of the unpacked tree, as
# validation/rank-n2/harness/build.sh does for 1.2.9:
#   precise  COLLIER's gfortran Release flags without -ffast-math
#            (-O2 -funroll-loops; the benchmark's headline, as no code in the
#            benchmark report uses -ffast-math)
#   stock    the shipped Release flags (-O2 -funroll-loops -ffast-math)
# Each variant gets BUILD_DIR/bin/collier_bench-<variant> and the global-cache reproducer
# BUILD_DIR/bin/collier_cache_repro-<variant> (collier_cache_repro.f90), compiled with the
# variant's flags and linked statically against its libcollier.a.
#
# Requirements: curl, sha256sum, gfortran (the repository dev shell) and cmake (taken
# from `nix shell nixpkgs#cmake` when not on PATH).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
OUT="${1:-${COLLIER_BENCH_BUILD:-$REPO/target/benchmark/collier-1.2.10}}"
URL="https://collier.hepforge.org/downloads/?f=collier-1.2.10.tar.gz"
SHA256="460624ea709211506b642343a7f6be41d3883ccb52598b14950f8e07c0262713"
JOBS="${JOBS:-16}"
command -v gfortran >/dev/null || { echo "gfortran not on PATH (enter the dev shell)" >&2; exit 1; }
if command -v cmake >/dev/null; then cm() { "$@"; }; else cm() { nix shell nixpkgs#cmake --command "$@"; }; fi
mkdir -p "$OUT/bin"
TARBALL="$OUT/download/collier-1.2.10.tar.gz"
if [ ! -f "$TARBALL" ]; then
  mkdir -p "$OUT/download"
  curl -sSfL -o "$TARBALL.part" "$URL"
  mv "$TARBALL.part" "$TARBALL"
fi
echo "$SHA256  $TARBALL" | sha256sum -c - >/dev/null || { echo "sha256 mismatch for $TARBALL" >&2; exit 1; }
SRC="$OUT/src/collier-COLLIER-1.2.10"
if [ ! -f "$SRC/src/COLLIER.F90" ]; then
  mkdir -p "$OUT/src"
  tar xzf "$TARBALL" -C "$OUT/src"
fi
grep -q 'version_cll="1.2.10"' "$SRC/src/collier_global.F90" || { echo "unexpected COLLIER version in $SRC" >&2; exit 1; }
RELEASE='set (opt_RELEASE "-O2 -funroll-loops -ffast-math -Wtabs")'
for v in precise stock; do
  d="$OUT/collier-$v"
  case $v in
    precise) flags="-O2 -funroll-loops" ;;
    stock)   flags="-O2 -funroll-loops -ffast-math" ;;
  esac
  if [ ! -f "$d/libcollier.a" ] && [ ! -f "$d/build/libcollier.a" ]; then
    rm -rf "$d"; mkdir -p "$d"
    cp -r "$SRC"/{CMakeLists.txt,COPYING,config,demos,src} "$d"/
    grep -qF "$RELEASE" "$d/CMakeLists.txt" || { echo "unexpected CMakeLists.txt release flags" >&2; exit 1; }
    [ $v = precise ] && sed -i "s|$RELEASE|set (opt_RELEASE \"-O2 -funroll-loops -Wtabs\")|" "$d/CMakeLists.txt"
    mkdir -p "$d/build"
    (cd "$d/build" && cm cmake -Dstatic=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER=gfortran .. > cmake.log 2>&1 \
       && cm make -j"$JOBS" collier > make.log 2>&1) || { echo "COLLIER $v build failed, see $d/build/*.log" >&2; exit 1; }
  fi
  lib="$d/build/libcollier.a"; [ -f "$lib" ] || lib="$d/libcollier.a"
  mods="$d/build/modules"; [ -d "$mods" ] || mods="$d/modules"
  # shellcheck disable=SC2086
  gfortran $flags -I"$mods" "$HERE/collier_bench.f90" "$lib" -o "$OUT/bin/collier_bench-$v"
  gfortran $flags -I"$mods" "$HERE/collier_cache_repro.f90" "$lib" -o "$OUT/bin/collier_cache_repro-$v"
  echo "built $OUT/bin/collier_bench-$v and collier_cache_repro-$v ($flags)"
done
