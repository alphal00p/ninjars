{
  description = "NinjaRS development and reference-validation toolchain";
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/02f5696b0e6097e589076d886b317b83ff0437d7";
    fenix.url = "github:nix-community/fenix/1fc6f36a23fa16e3740987957266f6e4b915c3cd";
    fenix.inputs.nixpkgs.follows = "nixpkgs";
    fenix.inputs.rust-analyzer-src.follows = "";
  };
  outputs = { nixpkgs, fenix, ... }: let
    system = "x86_64-linux";
    pkgs = import nixpkgs { inherit system; };
    rust = fenix.packages.${system}.stable.withComponents [
      "cargo" "rustc" "rust-std" "rustfmt" "clippy"
    ];
  in {
    devShells.${system}.default = pkgs.mkShell {
      packages = [ rust ] ++ (with pkgs; [
        gcc gfortran gnumake pkg-config gmp mpfr libmpc
        autoconf automake libtool perl m4 python3 gh git
      ]);
      RUST_BACKTRACE = "1";
    };
  };
}
