#!/usr/bin/env bash
exec nix shell --impure --expr '(builtins.getFlake "nixpkgs").legacyPackages.x86_64-linux.python3.withPackages (p: [p.sympy p.mpmath p.gmpy2])' --command python3 "$@"
