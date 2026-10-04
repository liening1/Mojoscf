"""Generate the specialised ERI kernel of mojoscf/_mojo/integrals.mojo.

The kernel ``eri_kernel_spec[LO, LI]`` is fully unrolled over Hermite indices
with Mojo ``comptime`` loops, but its register blocking (eight named SIMD
accumulators split between Hermite indices and partial sums) is written out
explicitly.  This script produces that repetitive code and replaces the text
between the BEGIN/END markers in integrals.mojo:

    python tools/gen_eri_kernel.py
"""
from __future__ import annotations

import pathlib
import re

TARGET = pathlib.Path(__file__).resolve().parents[1] / "mojoscf" / "_mojo" / "integrals.mojo"
BEGIN = "# BEGIN GENERATED eri_kernel_spec (tools/gen_eri_kernel.py)\n"
END = "# END GENERATED eri_kernel_spec\n"

NACC = 8  # SIMD accumulators per block


def ind(n: int) -> str:
    return " " * n


def inner_block(rr: str, rs: str, base: int, lanes: bool = False) -> list[str]:
    """Inner transform of one primitive quartet into tb.

    T[ho] += sum_hi R[ho + hi] E[hi]: blocks of nbh <= 8 outer Hermite indices,
    each with S = 8 // nbh partial sums over the inner index (hi % S), so that
    there are always eight independent FMA chains.  ``rr`` points at R of the
    primitive quartet, ``rs`` is the distance between consecutive R values.
    """
    b = base
    out = []
    out.append(f"{ind(b)}for iv in range(niv):")
    if not lanes:
        out.append(f"{ind(b + 4)}var e = eb.unsafe_offset(iv * W)")
    out.append(f"{ind(b + 4)}var tbo = iv * W")
    out.append(f"{ind(b + 4)}comptime for blk in range((NHO + {NACC - 1}) // {NACC}):")
    c = b + 8
    out.append(f"{ind(c)}comptime h0 = blk * {NACC}")
    out.append(f"{ind(c)}comptime nbh = min({NACC}, NHO - h0)")
    out.append(f"{ind(c)}comptime S = {NACC} // nbh if nbh < 3 else (2 if nbh < 5 else 1)")
    for k in range(NACC):
        out.append(f"{ind(c)}var a{k} = SIMD[DType.float64, W](0.0)")
    # accumulator k belongs to outer index h0 + k // S and partial sum k % S
    for k in range(NACC):
        out.append(f"{ind(c)}comptime if {k} % S == 0 and {k} // S < nbh:")
        out.append(f"{ind(c + 4)}a{k} = tb.unsafe_load[width=W]((h0 + {k} // S) * si + tbo)")
    if lanes:
        # all lanes of the batch accumulate into the same registers before the store
        out.append(f"{ind(c)}for j in range(nb):")
        c2 = c + 4
        out.append(f"{ind(c2)}var rr = rv.unsafe_offset(j)")
        out.append(f"{ind(c2)}var e = ei.unsafe_offset((ki0 + j) * NHI * si + iv * W)")
    else:
        c2 = c
    out.append(f"{ind(c2)}comptime for hi in range(NHI):")
    d = c2 + 4
    out.append(f"{ind(d)}var ev = e.unsafe_load[width=W](hi * si)")
    for k in range(NACC):
        out.append(f"{ind(d)}comptime if {k} // S < nbh and hi % S == {k} % S:")
        out.append(f"{ind(d + 4)}comptime i{k} = herm_sum(h0 + {k} // S, hi) * {rs}")
        out.append(f"{ind(d + 4)}a{k} += ev * {rr}[unsafe_offset=i{k}]")
    # combine the partial sums and store: S is 1, 2, 4 or 8
    out.append(f"{ind(c)}comptime if S == 1:")
    for k in range(NACC):
        out.append(f"{ind(c + 4)}comptime if {k} < nbh:")
        out.append(f"{ind(c + 8)}tb.unsafe_store((h0 + {k}) * si + tbo, a{k})")
    out.append(f"{ind(c)}elif S == 2:")
    for r in range(NACC // 2):
        out.append(f"{ind(c + 4)}comptime if {r} < nbh:")
        out.append(f"{ind(c + 8)}tb.unsafe_store((h0 + {r}) * si + tbo, a{2 * r} + a{2 * r + 1})")
    out.append(f"{ind(c)}elif S == 4:")
    for r in range(NACC // 4):
        out.append(f"{ind(c + 4)}comptime if {r} < nbh:")
        out.append(
            f"{ind(c + 8)}tb.unsafe_store((h0 + {r}) * si + tbo, (a{4 * r} + a{4 * r + 1}) + (a{4 * r + 2} + a{4 * r + 3}))"
        )
    out.append(f"{ind(c)}else:")
    out.append(f"{ind(c + 4)}tb.unsafe_store(h0 * si + tbo, ((a0 + a1) + (a2 + a3)) + ((a4 + a5) + (a6 + a7)))")
    return out


def recursion(vec: bool, base: int) -> list[str]:
    """Hermite Coulomb recursion, levels L .. 0, level n in buffer (n % 2).

    Scalar form: rs[(n % 2) NHL + h]; vector form: rv[((n % 2) NHL + h) W + lane].
    The starting values (level n, index 0) come from fbs[n] / fbv[n W].
    """
    b = base
    out = []
    if vec:
        ld = lambda i: f"rv.unsafe_load[width=W](({i}) * W)"
        st = lambda i, v: f"rv.unsafe_store(({i}) * W, {v})"
        f0 = lambda n: f"fbv.unsafe_load[width=W]({n} * W)"
    else:
        ld = lambda i: f"rs[unsafe_offset={i}]"
        st = lambda i, v: f"rs[unsafe_offset={i}] = {v}"
        f0 = lambda n: f"fbs[unsafe_offset={n}]"
    out.append(f"{ind(b)}{st('(L % 2) * NHL', f0('L'))}")
    out.append(f"{ind(b)}comptime for nn in range(L):")
    c = b + 4
    out.append(f"{ind(c)}comptime n = L - 1 - nn")
    out.append(f"{ind(c)}comptime cur = (n % 2) * NHL")
    out.append(f"{ind(c)}comptime prv = ((n + 1) % 2) * NHL")
    out.append(f"{ind(c)}{st('cur', f0('n'))}")
    out.append(f"{ind(c)}comptime for h in range(1, nherm(L - n)):")
    d = c + 4
    out.append(f"{ind(d)}comptime t = herm_t(h)")
    out.append(f"{ind(d)}comptime u = herm_u(h)")
    out.append(f"{ind(d)}comptime v = herm_v(h)")
    for kw, var, coord, idx1, idx2 in (
        ("if t > 0", "t", "x", "herm_index(t - 1, u, v)", "herm_index(t - 2, u, v)"),
        ("elif u > 0", "u", "y", "herm_index(t, u - 1, v)", "herm_index(t, u - 2, v)"),
        ("else", "v", "z", "herm_index(t, u, v - 1)", "herm_index(t, u, v - 2)"),
    ):
        out.append(f"{ind(d)}comptime {kw}:" if kw.startswith("if") else f"{ind(d)}{kw}:")
        e = d + 4
        out.append(f"{ind(e)}comptime i1 = {idx1}")
        out.append(f"{ind(e)}var val = {coord} * {ld('prv + i1')}")
        out.append(f"{ind(e)}comptime if {var} > 1:")
        out.append(f"{ind(e + 4)}comptime i2 = {idx2}")
        out.append(f"{ind(e + 4)}val += {ld('prv + i2')} * Float64({var} - 1)")
        out.append(f"{ind(e)}{st('cur + h', 'val')}")
    return out


def boys_lane(dst: str, base: int) -> list[str]:
    """F_n(tt) for n = 0..L into dst(n) (a statement template)."""
    b = base
    out = []
    out.append(f"{ind(b)}if tt < BoysTable.TMAX:")
    c = b + 4
    out += [
        f"{ind(c)}var k = Int(tt * BoysTable.INV_DT + 0.5)",
        f"{ind(c)}var dt = Float64(k) * BoysTable.DT - tt",
        f"{ind(c)}var row = btab.unsafe_offset(k * NR)",
        f"{ind(c)}var c1 = dt",
        f"{ind(c)}var c2 = c1 * dt * 0.5",
        f"{ind(c)}var c3 = c2 * dt * 0.3333333333333333",
        f"{ind(c)}var c4 = c3 * dt * 0.25",
        f"{ind(c)}var c5 = c4 * dt * 0.2",
        f"{ind(c)}var c6 = c5 * dt * 0.16666666666666666",
        f"{ind(c)}var c7 = c6 * dt * 0.14285714285714285",
        f"{ind(c)}comptime for n in range(L + 1):",
        f"{ind(c + 4)}var fv = (",
        f"{ind(c + 8)}row[unsafe_offset=n] + c1 * row[unsafe_offset=n + 1] + c2 * row[unsafe_offset=n + 2]",
        f"{ind(c + 8)}+ c3 * row[unsafe_offset=n + 3] + c4 * row[unsafe_offset=n + 4] + c5 * row[unsafe_offset=n + 5]",
        f"{ind(c + 8)}+ c6 * row[unsafe_offset=n + 6] + c7 * row[unsafe_offset=n + 7]",
        f"{ind(c + 4)})",
        f"{ind(c + 4)}{dst('n', 'fv')}",
    ]
    out.append(f"{ind(b)}else:")
    out += [
        f"{ind(c)}var fv = 0.5 * sqrt(PI / tt)",
        f"{ind(c)}var et = 0.0",
        f"{ind(c)}comptime if L > 0:",
        f"{ind(c + 4)}et = exp(-tt)",
        f"{ind(c)}var inv2t = 0.5 / tt",
        f"{ind(c)}comptime for n in range(L + 1):",
        f"{ind(c + 4)}{dst('n', 'fv')}",
        f"{ind(c + 4)}fv = (Float64(2 * n + 1) * fv - et) * inv2t",
    ]
    return out


def outer_step() -> list[str]:
    out = []
    A = out.append
    A("        # outer transform: M[o] += sum_ho (-1)^{|ho|} E_O[ko][ho][o] T[ho]; blocks of four o")
    A("        # and two SIMD vectors give eight independent FMA chains")
    A("        var eob = eo.unsafe_offset(ko * NHO * so)")
    A("        var o = 0")
    A("        while o + 4 <= no:")
    A("            var er = eob.unsafe_offset(o)")
    A("            var iv = 0")
    A("            while iv + 2 <= niv:")
    A("                var mb = m.unsafe_offset(o * si + iv * W)")
    for r in range(4):
        for c in range(2):
            A(f"                var a{r}{c} = mb.unsafe_load[width=W]({r} * si + {c} * W)")
    A("                comptime for ho in range(NHO):")
    A("                    comptime odd = herm_odd(ho)")
    A("                    var t0 = tb.unsafe_load[width=W](ho * si + iv * W)")
    A("                    var t1 = tb.unsafe_load[width=W](ho * si + iv * W + W)")
    A("                    var c = er.unsafe_offset(ho * so)")
    for r in range(4):
        A(f"                    var c{r} = c[unsafe_offset={r}]")
    A("                    comptime if odd:")
    for r in range(4):
        for c in range(2):
            A(f"                        a{r}{c} -= t{c} * c{r}")
    A("                    else:")
    for r in range(4):
        for c in range(2):
            A(f"                        a{r}{c} += t{c} * c{r}")
    for r in range(4):
        for c in range(2):
            A(f"                mb.unsafe_store({r} * si + {c} * W, a{r}{c})")
    A("                iv += 2")
    A("            if iv < niv:")
    A("                var mb = m.unsafe_offset(o * si + iv * W)")
    for r in range(4):
        A(f"                var a{r} = mb.unsafe_load[width=W]({r} * si)")
        A(f"                var b{r} = SIMD[DType.float64, W](0.0)")
    A("                comptime for ho in range(NHO):")
    A("                    comptime odd = herm_odd(ho)")
    A("                    var t0 = tb.unsafe_load[width=W](ho * si + iv * W)")
    A("                    var c = er.unsafe_offset(ho * so)")
    A("                    comptime if ho % 2 == 0:")
    A("                        comptime if odd:")
    for r in range(4):
        A(f"                            a{r} -= t0 * c[unsafe_offset={r}]")
    A("                        else:")
    for r in range(4):
        A(f"                            a{r} += t0 * c[unsafe_offset={r}]")
    A("                    else:")
    A("                        comptime if odd:")
    for r in range(4):
        A(f"                            b{r} -= t0 * c[unsafe_offset={r}]")
    A("                        else:")
    for r in range(4):
        A(f"                            b{r} += t0 * c[unsafe_offset={r}]")
    for r in range(4):
        A(f"                mb.unsafe_store({r} * si, a{r} + b{r})")
    A("            o += 4")
    A("        while o < no:")
    A("            var er = eob.unsafe_offset(o)")
    A("            for iv in range(niv):")
    A("                var mb = m.unsafe_offset(o * si + iv * W)")
    for k in range(4):
        A(f"                var a{k} = SIMD[DType.float64, W](0.0)")
    A("                a0 = mb.unsafe_load[width=W](0)")
    A("                comptime for ho in range(NHO):")
    A("                    comptime odd = herm_odd(ho)")
    A("                    var tv = tb.unsafe_load[width=W](ho * si + iv * W)")
    for k in range(4):
        kw = "comptime if" if k == 0 else "elif"
        A(f"                    {kw} ho % 4 == {k}:")
        A("                        comptime if odd:")
        A(f"                            a{k} -= tv * er[unsafe_offset=ho * so]")
        A("                        else:")
        A(f"                            a{k} += tv * er[unsafe_offset=ho * so]")
    A("                mb.unsafe_store(0, (a0 + a1) + (a2 + a3))")
    A("            o += 1")
    return out


def kernel() -> str:
    out = []
    A = out.append
    A('''def eri_kernel_spec[LO: Int, LI: Int](
    nop: Int, po: F64Ptr, eo: F64Ptr, so: Int, no: Int,
    nip: Int, pin: F64Ptr, ei: F64Ptr, si: Int,
    btab: F64Ptr, m: F64Ptr, rbuf: F64Ptr,
):
    """(outer|inner) for Hermite degrees LO, LI known at compile time; M[o][i] += result.

    Everything except the loops over primitives, inner SIMD vectors and outer
    components is unrolled.  When at least VMIN inner primitives remain they
    are processed W at a time: prefactors, Boys arguments and the Hermite
    Coulomb recursion are evaluated as SIMD vectors over the W primitive
    quartets (only the table look-up of the Boys function runs per lane) and
    R is stored as ``rbuf[h W + lane]``; otherwise one primitive quartet at a
    time with scalar arithmetic.  Either way the transforms take R as an
    embedded broadcast operand.  ``rbuf`` must be 64-byte aligned.
    """
    comptime NHO = nherm(LO)
    comptime NHI = nherm(LI)
    comptime L = LO + LI
    comptime NHL = nherm(L)
    comptime NR = BoysTable.NROWS
    var niv = si // W
    var opad = padded(nop)
    var ipad = padded(nip)
    var tb = stack_allocation[NHO * SIMAX, Float64, alignment=64]()
    var rv = rbuf                                   # vector path: two recursion levels, [h][lane]
    var fbv = rbuf.unsafe_offset(2 * NHL * W)       # vector path: gathered table rows, then scaled F_n
    var tv = rbuf.unsafe_offset((2 * NHL + L + 8) * W)   # vector path: F_n
    var rs = rbuf.unsafe_offset((2 * NHL + 2 * L + 9) * W)   # scalar path: two recursion levels
    var fbs = rs.unsafe_offset(2 * NHL)                        # scalar path: scaled Boys values
    for ko in range(nop):
        var p = po[unsafe_offset=ko]
        var ip = po[unsafe_offset=opad + ko]
        var px = po[unsafe_offset=2 * opad + ko]
        var py = po[unsafe_offset=3 * opad + ko]
        var pz = po[unsafe_offset=4 * opad + ko]
        for x in range(NHO * si):
            tb[unsafe_offset=x] = 0.0
        var ki0 = 0
        while ki0 < nip:
            var nb = min(W, nip - ki0)
            if nb >= VMIN:
                var q = pin.unsafe_load[width=W](ki0)
                var x = pin.unsafe_load[width=W](2 * ipad + ki0) - px
                var y = pin.unsafe_load[width=W](3 * ipad + ki0) - py
                var z = pin.unsafe_load[width=W](4 * ipad + ki0) - pz
                var inv_s = 1.0 / (q + p)
                var alpha = q * p * inv_s
                var pref = pin.unsafe_load[width=W](ipad + ki0) * (TWO_PI_52 * ip) * sqrt(inv_s)
                var tt = alpha * (x * x + y * y + z * z)
                # Boys function of all lanes: Taylor expansion about the nearest grid point
                # (table rows gathered per lane), blended with the asymptotic form for T >= TMAX
                var near = tt.lt(BoysTable.TMAX)
                var ttc = min(tt, SIMD[DType.float64, W](BoysTable.TMAX))
                var kk = (ttc * BoysTable.INV_DT + 0.5).cast[DType.int64]()
                var dt = kk.cast[DType.float64]() * BoysTable.DT - ttc
                var off = kk * NR
                var c1 = dt
                var c2 = c1 * dt * 0.5
                var c3 = c2 * dt * 0.3333333333333333
                var c4 = c3 * dt * 0.25
                var c5 = c4 * dt * 0.2
                var c6 = c5 * dt * 0.16666666666666666
                var c7 = c6 * dt * 0.14285714285714285
                comptime for mm in range(L + 8):
                    fbv.unsafe_store(mm * W, btab.unsafe_gather(off + SIMD[DType.int64, W](mm)))
                comptime for n in range(L + 1):
                    var fv = (
                        fbv.unsafe_load[width=W](n * W) + c1 * fbv.unsafe_load[width=W]((n + 1) * W)
                        + c2 * fbv.unsafe_load[width=W]((n + 2) * W) + c3 * fbv.unsafe_load[width=W]((n + 3) * W)
                        + c4 * fbv.unsafe_load[width=W]((n + 4) * W) + c5 * fbv.unsafe_load[width=W]((n + 5) * W)
                        + c6 * fbv.unsafe_load[width=W]((n + 6) * W) + c7 * fbv.unsafe_load[width=W]((n + 7) * W)
                    )
                    tv.unsafe_store(n * W, fv)
                if not near.reduce_and():
                    # vector exp is accurate to ~1e-11 relative, but exp(-T) < 2.4e-16 here and only
                    # enters as a correction, so the result keeps full precision
                    var tts = max(tt, SIMD[DType.float64, W](BoysTable.TMAX))
                    var fa = sqrt(PI / tts) * 0.5
                    var et = SIMD[DType.float64, W](0.0)
                    comptime if L > 0:
                        et = std_exp(-tts)
                    var inv2t = 0.5 / tts
                    comptime for n in range(L + 1):
                        tv.unsafe_store(n * W, near.select(tv.unsafe_load[width=W](n * W), fa))
                        fa = (fa * Float64(2 * n + 1) - et) * inv2t
                # level n of the recursion starts from pref (-2 alpha)^n F_n(T)
                var m2a = alpha * -2.0
                var g = pref
                comptime for n in range(L + 1):
                    fbv.unsafe_store(n * W, tv.unsafe_load[width=W](n * W) * g)
                    g = g * m2a''')
    out += recursion(True, 16)
    A("                # inner transforms of the W lanes")
    out += inner_block("rr", "W", 16, lanes=True)
    A('''                ki0 += nb
            else:
                var q = pin[unsafe_offset=ki0]
                var x = pin[unsafe_offset=2 * ipad + ki0] - px
                var y = pin[unsafe_offset=3 * ipad + ki0] - py
                var z = pin[unsafe_offset=4 * ipad + ki0] - pz
                var inv_s = 1.0 / (p + q)
                var alpha = p * q * inv_s
                var pref = TWO_PI_52 * ip * pin[unsafe_offset=ipad + ki0] * sqrt(inv_s)
                var tt = alpha * (x * x + y * y + z * z)''')
    out += boys_lane(lambda n, v: f"fbs[unsafe_offset={n}] = {v}", 16)
    A('''                var m2a = -2.0 * alpha
                var g = pref
                comptime for n in range(L + 1):
                    fbs[unsafe_offset=n] = fbs[unsafe_offset=n] * g
                    g *= m2a''')
    out += recursion(False, 16)
    A("                var eb = ei.unsafe_offset(ki0 * NHI * si)")
    out += inner_block("rs", "1", 16)
    A("                ki0 += 1")
    out += outer_step()
    return "\n".join(out) + "\n"


def main():
    text = TARGET.read_text()
    i = text.index(BEGIN) + len(BEGIN)
    j = text.index(END)
    TARGET.write_text(text[:i] + kernel() + "\n\n" + text[j:])
    print(f"wrote eri_kernel_spec into {TARGET}")


if __name__ == "__main__":
    main()
