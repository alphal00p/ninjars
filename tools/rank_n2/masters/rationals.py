# Rational parts R = -(r-1)! * Res_eps I_N^{4+2r-2eps}[F] of mu^{2r} integrals,
# I_N^D = (-1)^N Gamma(N - D/2 + k?) ... via Feynman parameters (tensor F = (q.e)^k with massless e).
import sympy as sp
eps = sp.symbols('eps')

def simplex_integral(expr, xs):
    # integrate over x_i >= 0, sum x = 1 (last eliminated)
    n = len(xs)
    sub = {xs[-1]: 1 - sum(xs[:-1])}
    e = sp.expand(expr.subs(sub))
    lims = []
    rest = 1
    for i, x in enumerate(xs[:-1]):
        pass
    # nested integration
    def nest(e, i, upper):
        if i == n - 1:
            return e
        x = xs[i]
        inner = nest(e, i + 1, upper - x)
        return sp.integrate(inner, (x, 0, upper))
    return sp.simplify(nest(e, 0, 1))

def rational(N, r, masses, V, tensor_power=0, edot=None):
    xs = sp.symbols(f'x0:{N}', nonnegative=True)
    # Delta = sum x m^2 - sum_{i<j} x_i x_j s_ij
    s = {}
    Delta = sum(x * m for x, m in zip(xs, masses))
    for i in range(N):
        for j in range(i + 1, N):
            Delta -= xs[i] * xs[j] * V[(i, j)]
    D = 4 + 2 * r - 2 * eps
    # scalar product (P.e) with P = sum x_i V_i, q = l - P -> (q.e)^k -> (-P.e)^k (massless e: no g terms)
    pe = -sum(x * ve for x, ve in zip(xs, edot)) if tensor_power else 1
    a = N - D / 2  # Gamma(a) * Delta^{-a}
    # pole of Gamma(a) where a = N-2-r+eps = -k + eps, k = r+2-N >= 0: Gamma(-k+eps) ~ (-1)^k/(k! eps)
    k = r + 2 - N
    if k < 0:
        return 0
    res_gamma = sp.Integer(-1) ** k / sp.factorial(k)
    integrand = (pe ** tensor_power) * Delta ** k
    I = simplex_integral(integrand, list(xs))
    res = (-1) ** N * res_gamma * I
    return sp.simplify(-sp.factorial(r - 1) * res)

m = sp.symbols('m0:4')
s = sp.symbols('s')
# checks against Ninja
print('box mu^4      ', rational(4, 2, m, {(i, j): sp.Symbol(f's{i}{j}') for i in range(4) for j in range(4)}))
print('tri mu^2      ', rational(3, 1, m[:3], {(i, j): sp.Symbol(f's{i}{j}') for i in range(3) for j in range(3)}))
print('tri mu^4      ', sp.expand(rational(3, 2, m[:3], {(i, j): sp.Symbol(f's{i}{j}') for i in range(3) for j in range(3)})))
print('bub mu^2      ', sp.factor(rational(2, 1, m[:2], {(0, 1): s})))
print('tad mu^2      ', rational(1, 1, m[:1], {}))
ke2 = sp.Symbol('ke2')
print('bub mu^2 w    ', sp.factor(rational(2, 1, m[:2], {(0, 1): s}, 1, [0, ke2])))
# new for rank n+2
print('box mu^6      ', sp.expand(rational(4, 3, m, {(i, j): sp.Symbol(f's{i}{j}') for i in range(4) for j in range(4)})))
print('bub mu^4      ', sp.expand(rational(2, 2, m[:2], {(0, 1): s})))
print('bub mu^2 w^2  ', sp.expand(rational(2, 1, m[:2], {(0, 1): s}, 2, [0, ke2])))
print('tri mu^6 (n/a)', sp.expand(rational(3, 3, m[:3], {(i, j): sp.Symbol(f's{i}{j}') for i in range(3) for j in range(3)})))
print('tad mu^4 (n/a)', rational(1, 2, m[:1], {}))
