"""tvm.py — time-value-of-money helpers for 16:540:580:90.

Concept-first: you call these; you do not derive them. Every function takes the
per-period interest rate i as a fraction (0.12 for 12%) and the number of
periods n. Run `python tvm.py` to execute the self-test.
"""
from __future__ import annotations
import numpy as np

def f_p(i, n): return (1 + i) ** n                     # future worth of a present sum
def p_f(i, n): return (1 + i) ** -n                    # present worth of a future sum
def a_p(i, n): return i * (1 + i) ** n / ((1 + i) ** n - 1) if i else 1 / n   # capital recovery
def p_a(i, n): return 1 / a_p(i, n)                    # present worth of a uniform series
def f_a(i, n): return ((1 + i) ** n - 1) / i if i else n
def a_f(i, n): return 1 / f_a(i, n)
def p_g(i, n):                                          # present worth of an arithmetic gradient (0, G, 2G, ...)
    return ((1 + i) ** n - i * n - 1) / (i ** 2 * (1 + i) ** n)
def p_geo(i, g, n, a1=1.0):
    """Present worth of a geometric series a1, a1(1+g), ... for n periods."""
    if abs(i - g) < 1e-12: return a1 * n / (1 + i)
    return a1 * (1 - ((1 + g) / (1 + i)) ** n) / (i - g)

def effective_rate(nominal_annual, m_per_year, periods_per_year=1):
    """Effective rate per `periods_per_year` block from a nominal annual rate compounded m times/yr."""
    return (1 + nominal_annual / m_per_year) ** (m_per_year / periods_per_year) - 1

def pw(cash_flows, i):
    """Present worth of cash flows indexed by period 0..n (list or array)."""
    cf = np.asarray(cash_flows, dtype=float)
    return float((cf * (1 + i) ** -np.arange(len(cf))).sum())

def aw(cash_flows, i):
    """Annual worth over the life of the cash-flow series."""
    n = len(cash_flows) - 1
    return pw(cash_flows, i) * a_p(i, n)

def irr(cash_flows, lo=-0.99, hi=10.0, tol=1e-9):
    """Internal rate of return by bisection; returns None if no sign change."""
    cf = np.asarray(cash_flows, dtype=float)
    f = lambda r: pw(cf, r)
    if f(lo) * f(hi) > 0: return None
    for _ in range(300):
        mid = (lo + hi) / 2
        if f(lo) * f(mid) <= 0: hi = mid
        else: lo = mid
        if hi - lo < tol: break
    return (lo + hi) / 2

def err(cash_flows, eps):
    """External rate of return: reinvest positives at eps, discount negatives at eps."""
    cf = np.asarray(cash_flows, dtype=float); n = len(cf) - 1
    neg = cf.clip(max=0); pos = cf.clip(min=0)
    pw_neg = -pw(neg, eps); fw_pos = float((pos * (1 + eps) ** (n - np.arange(n + 1))).sum())
    return (fw_pos / pw_neg) ** (1 / n) - 1

def incremental_irr(cf_low, cf_high):
    """IRR on the increment (higher-first-cost minus lower-first-cost alternative)."""
    return irr(np.asarray(cf_high, float) - np.asarray(cf_low, float))

def _selftest():
    assert abs(p_a(0.10, 5) - 3.7908) < 1e-3
    assert abs(a_p(0.08, 10) - 0.14903) < 1e-4
    assert abs(f_a(0.06, 4) - 4.3746) < 1e-3
    assert abs(p_g(0.10, 5) - 6.8618) < 1e-3
    assert abs(effective_rate(0.12, 4) - 0.12551) < 1e-4
    assert abs(irr([-100, 60, 60]) - 0.13066) < 1e-4
    print("tvm.py self-test passed")

if __name__ == "__main__":
    _selftest()
