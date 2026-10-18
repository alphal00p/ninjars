#!/usr/bin/env bash
# Run python3 with sympy/mpmath/numpy/gmpy2 (the documented 'nix shell nixpkgs#python3 nixpkgs#python3Packages.sympy ...'
# does not expose sympy to python 3.14 here, so we use python3.withPackages).
exec nix shell --impure --expr '(builtins.getFlake "nixpkgs").legacyPackages.x86_64-linux.python3.withPackages (p: [p.sympy p.mpmath p.numpy p.gmpy2])' --command python3 "$@"
