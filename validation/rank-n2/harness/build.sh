#!/usr/bin/env bash
# Build COLLIER 1.2.9 and the tensor driver used by the rank n+2 reference harness.
#
#   validation/rank-n2/harness/build.sh [BUILD_DIR]
#
# BUILD_DIR defaults to $COLLIER_BUILD or <repo>/target/rank-n2-collier (gitignored).
# Three COLLIER variants are built from an unmodified copy of the source tree:
#   quad     COLLIER's gfortran Release flags without -ffast-math, plus -freal-8-real-16
#            (every REAL(8)/COMPLEX(8) entity and d0 literal becomes kind 16). The
#            reference build.
#   precise  the Release flags without -ffast-math (binary64 diagnostics).
#   stock    the shipped Release flags (-O2 -funroll-loops -ffast-math), for comparison.
# Each gets its driver BUILD_DIR/bin/collier_tensor_driver-<variant>; the quad driver
# prints 35 significant digits (ES44.34E4) instead of 17.
#
# Requirements: gfortran (the repository dev shell, e.g. `source devenv.sh` or
# `nix develop`) and cmake (taken from `nix shell nixpkgs#cmake` when not on PATH).
# Source: COLLIER_SRC, default /common/dev/MG5_aMC_v3_7_0/HEPTools/collier/COLLIER-1.2.9
# (the copy MG5_aMC v3.7.0 installs; upstream tarball collier-1.2.9.tar.gz from
# https://collier.hepforge.org gives the same src/ tree).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../../.." && pwd)"
OUT="${1:-${COLLIER_BUILD:-$REPO/target/rank-n2-collier}}"
SRC="${COLLIER_SRC:-/common/dev/MG5_aMC_v3_7_0/HEPTools/collier/COLLIER-1.2.9}"
JOBS="${JOBS:-16}"
[ -f "$SRC/src/COLLIER.F90" ] || { echo "COLLIER source not found at $SRC" >&2; exit 1; }
command -v gfortran >/dev/null || { echo "gfortran not on PATH (enter the dev shell)" >&2; exit 1; }
if command -v cmake >/dev/null; then cm() { "$@"; }; else cm() { nix shell nixpkgs#cmake --command "$@"; }; fi
mkdir -p "$OUT/bin"
RELEASE='set (opt_RELEASE "-O2 -funroll-loops -ffast-math -Wtabs")'
for v in quad precise stock; do
  d="$OUT/collier-$v"
  if [ ! -f "$d/libcollier.a" ] && [ ! -f "$d/build/libcollier.a" ]; then
    rm -rf "$d"; mkdir -p "$d"
    cp -r "$SRC"/{CMakeLists.txt,COPYING,config,demos,src,collierConfig.cmake,collierConfigVersion.cmake} "$d"/
    grep -qF "$RELEASE" "$d/CMakeLists.txt" || { echo "unexpected CMakeLists.txt release flags" >&2; exit 1; }
    case $v in
      quad)    sed -i "s|$RELEASE|set (opt_RELEASE \"-O2 -funroll-loops -Wtabs -freal-8-real-16\")|" "$d/CMakeLists.txt" ;;
      precise) sed -i "s|$RELEASE|set (opt_RELEASE \"-O2 -funroll-loops -Wtabs\")|" "$d/CMakeLists.txt" ;;
      stock)   ;;
    esac
    mkdir -p "$d/build"
    (cd "$d/build" && cm cmake -Dstatic=ON -DCMAKE_BUILD_TYPE=Release -DCMAKE_Fortran_COMPILER=gfortran .. > cmake.log 2>&1 \
       && cm make -j"$JOBS" > make.log 2>&1) || { echo "COLLIER $v build failed, see $d/build/*.log" >&2; exit 1; }
  fi
  lib="$d/build/libcollier.a"; [ -f "$lib" ] || lib="$d/libcollier.a"
  mods="$d/build/modules"; [ -d "$mods" ] || mods="$d/modules"
  if [ $v = quad ]; then
    sed 's/ES25.16E3/ES44.34E4/g' "$HERE/collier_tensor_driver.f90" > "$OUT/collier_tensor_driver_quad.f90"
    gfortran -O2 -freal-8-real-16 -I"$mods" "$OUT/collier_tensor_driver_quad.f90" "$lib" -o "$OUT/bin/collier_tensor_driver-$v"
  else
    gfortran -O2 -I"$mods" "$HERE/collier_tensor_driver.f90" "$lib" -o "$OUT/bin/collier_tensor_driver-$v"
  fi
  echo "built $OUT/bin/collier_tensor_driver-$v"
done
