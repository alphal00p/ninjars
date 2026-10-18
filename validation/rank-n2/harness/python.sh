#!/usr/bin/env bash
# Python 3 with mpmath and gmpy2 (nixpkgs), as used for every script in this directory.
# Any python3 >= 3.10 with mpmath (and optionally gmpy2) works as well.
exec nix shell --impure --expr '(builtins.getFlake "nixpkgs").legacyPackages.${builtins.currentSystem}.python3.withPackages (p: [p.mpmath p.gmpy2])' --command python3 -I "$@"
