# Symbolica library licensing

NinjaRS embeds a Symbolica library key issued for the Rust crate `ninjars`.
Its registration uses Symbolica's `set_library_key!` API, which verifies the
signed package name against the registering crate. The key's stated expiry
is 1 March 2028; continued authorization follows Symbolica's validation and
renewal rules.

While the key is accepted and valid, users do not need a personal Symbolica
key for the covered NinjaRS operations. This includes symbolic preparation,
generic reduction and oneloopmaster evaluations performed as part of those
operations, including use through the supported MadLoop interface. It does
not authorize standalone oneloopmaster use or unrelated Symbolica work in
the calling application. Symbolica retains its own license terms.

The persistent `LibraryUnlock` is private to NinjaRS. Internal workspace
crates and supported interfaces acquire a thread-bound scope through the
hidden `ninjars::licensing::unlock_for_ninjars_operation()` helper before
performing NinjaRS work. This helper is an internal integration API, not a
general authorization service for application code. The guard must cover
initialization as well as evaluation and must be dropped before returning
to unrelated caller work. Synchronous dependency calls and numerator
callbacks inherit the scope. Each NinjaRS-owned worker acquires its own
guard; Symbolica-dependent jobs may also propagate the current scope through
Symbolica's `InheritedLibraryUnlock` API.

The generic amplitude constructor, reduction entry point, and standalone
residue-reducer/integrator evaluation entry points hold guards.
The Symbolica adapters guard their preparation and master-evaluation paths.
The CLI acquires guards inside command workers, and the MadLoop adapter
holds them on the threads that prepare and evaluate its generic kernels.
The separate fast binary64 reduction engine does not add a guard to its hot
evaluation loop.

Registration failures do not print or expose the key. An expired or revoked
key may yield an inert guard; successful registration alone is not proof
that an operation is currently authorized. In either case Symbolica's usual
personal-license or restricted-mode behavior remains available, and the
user's `SYMBOLICA_LICENSE` setting is never changed by the library.

The `ninjars` integration test `library_license` starts a fresh subprocess
without `SYMBOLICA_LICENSE`. It checks that nested operations and simultaneous
worker threads can access Symbolica, that a panic drops the scope, and that
the unrelated caller is unlicensed after each operation. Run it with:

```sh
cargo test --locked -p ninjars --test library_license
```

Tests or applications that invoke Symbolica directly outside NinjaRS's scope
still need their own applicable authorization. The package key must not be
extracted or reused for such calls.
