// C++ kernels for nanocrystal-builder.
//
// pair_counts: coordination numbers from per-species-pair distance cutoffs
// with a cell list (uniform grid), the hot loop of the charge balance.
// Pairs whose distance lies within `eps` of their cutoff are not decided
// here: they are returned so the caller can settle them with the same
// distance routine as the reference implementation (scipy's cKDTree), which
// keeps results identical to it to the last bit.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

namespace nb = nanobind;

using Points = nb::ndarray<const double, nb::shape<-1, 3>, nb::c_contig, nb::device::cpu>;
using IntVec = nb::ndarray<const int32_t, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using RealVec = nb::ndarray<const double, nb::shape<-1>, nb::c_contig, nb::device::cpu>;
using RealMat = nb::ndarray<const double, nb::shape<-1, -1>, nb::c_contig, nb::device::cpu>;

template <typename T>
static nb::ndarray<nb::numpy, T> to_numpy(std::vector<T> &&v, std::initializer_list<size_t> shape) {
    auto *heap = new std::vector<T>(std::move(v));
    nb::capsule owner(heap, [](void *p) noexcept { delete static_cast<std::vector<T> *>(p); });
    return nb::ndarray<nb::numpy, T>(heap->data(), shape, owner);
}

// Uniform grid over the points' bounding box; cell edge >= the largest cutoff,
// so every neighbour of an atom lies in its own or the 26 adjacent cells.
struct CellList {
    double lo[3];
    double h;
    int64_t n[3];
    std::vector<int64_t> start;   // per cell, into `order`
    std::vector<int64_t> order;   // atom ids sorted by cell
    std::vector<int64_t> cell_of; // per atom

    CellList(const double *p, int64_t N, double rcut) {
        double hi[3];
        for (int k = 0; k < 3; ++k) { lo[k] = p[k]; hi[k] = p[k]; }
        for (int64_t i = 0; i < N; ++i)
            for (int k = 0; k < 3; ++k) {
                lo[k] = std::min(lo[k], p[3 * i + k]);
                hi[k] = std::max(hi[k], p[3 * i + k]);
            }
        h = std::max(rcut, 1e-6);
        // Cap the cell count (memory) for sparse or very wide point sets.
        const double cap = std::max<double>(1e6, 8.0 * double(N));
        for (;;) {
            double cells = 1.0;
            for (int k = 0; k < 3; ++k) cells *= std::floor((hi[k] - lo[k]) / h) + 1.0;
            if (cells <= cap) break;
            h *= 1.25;
        }
        for (int k = 0; k < 3; ++k) n[k] = int64_t(std::floor((hi[k] - lo[k]) / h)) + 1;
        const int64_t ncell = n[0] * n[1] * n[2];
        cell_of.resize(N);
        start.assign(ncell + 1, 0);
        for (int64_t i = 0; i < N; ++i) {
            int64_t c[3];
            for (int k = 0; k < 3; ++k)
                c[k] = std::min<int64_t>(n[k] - 1, int64_t((p[3 * i + k] - lo[k]) / h));
            cell_of[i] = (c[0] * n[1] + c[1]) * n[2] + c[2];
            ++start[cell_of[i] + 1];
        }
        for (int64_t c = 0; c < ncell; ++c) start[c + 1] += start[c];
        order.resize(N);
        std::vector<int64_t> fill(start.begin(), start.end() - 1);
        for (int64_t i = 0; i < N; ++i) order[fill[cell_of[i]]++] = i;
    }

    // Cells that can hold a point within h of x, for any x (inside the box
    // or not). Atoms on the top face are stored in the last cell, so the
    // range always reaches it; extra cells only cost distance checks.
    template <typename F>
    void for_cells_near(const double *x, F &&f) const {
        int64_t lo_c[3], hi_c[3];
        for (int k = 0; k < 3; ++k) {
            const double fc = std::floor((x[k] - lo[k]) / h);
            const int64_t c = fc < -2.0 ? -2 : (fc > double(n[k]) + 2.0 ? n[k] + 2 : int64_t(fc));
            lo_c[k] = std::max<int64_t>(0, std::min(c - 1, n[k] - 1));
            hi_c[k] = std::min(n[k] - 1, c + 1);
            if (lo_c[k] > hi_c[k]) return;
        }
        for (int64_t a = lo_c[0]; a <= hi_c[0]; ++a)
            for (int64_t b = lo_c[1]; b <= hi_c[1]; ++b)
                for (int64_t c = lo_c[2]; c <= hi_c[2]; ++c)
                    f((a * n[1] + b) * n[2] + c);
    }

    template <typename F>
    void for_neighbor_cells(int64_t cell, F &&f) const {
        const int64_t cz = cell % n[2], cy = (cell / n[2]) % n[1], cx = cell / (n[1] * n[2]);
        for (int64_t x = std::max<int64_t>(0, cx - 1); x <= std::min(n[0] - 1, cx + 1); ++x)
            for (int64_t y = std::max<int64_t>(0, cy - 1); y <= std::min(n[1] - 1, cy + 1); ++y)
                for (int64_t z = std::max<int64_t>(0, cz - 1); z <= std::min(n[2] - 1, cz + 1); ++z)
                    f((x * n[1] + y) * n[2] + z);
    }
};

// For every atom i, the number of atoms j != i with d(i, j) <= cutoff[type i, type j]
// and, when `bipartite`, q_i * q_j < 0 or q_i == 0.  Returns (counts, undecided)
// where `undecided` lists the (i, j) directions with |d - cutoff| <= eps; they
// are not included in `counts`.
static nb::tuple pair_counts(Points pts, IntVec type_id, RealVec charge, RealMat cutoff,
                             bool bipartite, double eps) {
    const int64_t N = int64_t(pts.shape(0));
    if (int64_t(type_id.shape(0)) != N || int64_t(charge.shape(0)) != N)
        throw nb::value_error("pts, type_id and charge must have the same length");
    if (cutoff.shape(0) != cutoff.shape(1))
        throw nb::value_error("cutoff must be square");
    const int64_t T = int64_t(cutoff.shape(0));
    const double *p = pts.data();
    const int32_t *t = type_id.data();
    const double *q = charge.data();
    const double *cut = cutoff.data();
    for (int64_t i = 0; i < N; ++i)
        if (t[i] < 0 || t[i] >= T) throw nb::value_error("type_id out of range");

    std::vector<int64_t> counts(N, 0);
    std::vector<int64_t> undecided;
    {
        nb::gil_scoped_release release;
        if (N > 0) {
            double rmax = 0.0;
            for (int64_t k = 0; k < T * T; ++k) rmax = std::max(rmax, cut[k]);
            const CellList cl(p, N, rmax + eps);

            auto visit = [&](int64_t i, int64_t j, double d) {
                if (bipartite && !(q[i] * q[j] < 0.0 || q[i] == 0.0)) return;
                const double c = cut[int64_t(t[i]) * T + t[j]];
                if (std::fabs(d - c) <= eps) {
                    undecided.push_back(i);
                    undecided.push_back(j);
                } else if (d < c) {
                    ++counts[i];
                }
            };
            for (int64_t i = 0; i < N; ++i) {
                cl.for_neighbor_cells(cl.cell_of[i], [&](int64_t cell) {
                    for (int64_t s = cl.start[cell]; s < cl.start[cell + 1]; ++s) {
                        const int64_t j = cl.order[s];
                        if (j <= i) continue;
                        const double dx = p[3 * i] - p[3 * j];
                        const double dy = p[3 * i + 1] - p[3 * j + 1];
                        const double dz = p[3 * i + 2] - p[3 * j + 2];
                        const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
                        if (d > rmax + eps) continue;
                        visit(i, j, d);
                        visit(j, i, d);
                    }
                });
            }
        }
    }
    const size_t M = undecided.size() / 2;
    return nb::make_tuple(to_numpy(std::move(counts), {size_t(N)}),
                          to_numpy(std::move(undecided), {M, 2}));
}

static inline double dist(const double *a, const double *b) {
    const double dx = a[0] - b[0], dy = a[1] - b[1], dz = a[2] - b[2];
    return std::sqrt(dx * dx + dy * dy + dz * dz);
}

static void check_types(const int32_t *t, int64_t N, int64_t T) {
    for (int64_t i = 0; i < N; ++i)
        if (t[i] < 0 || t[i] >= T) throw nb::value_error("type_id out of range");
}

// pair_counts restricted to the atoms within `seed_radius` of any seed point
// (a superset of the atoms whose count an edit at the seeds can change).
// Returns (atoms, counts, undecided) with the same rules as pair_counts.
static nb::tuple pair_counts_local(Points pts, IntVec type_id, RealVec charge, RealMat cutoff,
                                   bool bipartite, double eps, Points seeds, double seed_radius) {
    const int64_t N = int64_t(pts.shape(0)), K = int64_t(seeds.shape(0));
    if (int64_t(type_id.shape(0)) != N || int64_t(charge.shape(0)) != N)
        throw nb::value_error("pts, type_id and charge must have the same length");
    if (cutoff.shape(0) != cutoff.shape(1)) throw nb::value_error("cutoff must be square");
    const int64_t T = int64_t(cutoff.shape(0));
    const double *p = pts.data(), *q = charge.data(), *cut = cutoff.data(), *sp = seeds.data();
    const int32_t *t = type_id.data();
    check_types(t, N, T);

    std::vector<int64_t> atoms, counts, undecided;
    {
        nb::gil_scoped_release release;
        if (N > 0 && K > 0) {
            double rmax = 0.0;
            for (int64_t k = 0; k < T * T; ++k) rmax = std::max(rmax, cut[k]);
            const CellList cl(p, N, std::max(rmax + eps, seed_radius));
            std::vector<char> hit(N, 0);
            for (int64_t s = 0; s < K; ++s)
                cl.for_cells_near(sp + 3 * s, [&](int64_t cell) {
                    for (int64_t k = cl.start[cell]; k < cl.start[cell + 1]; ++k) {
                        const int64_t j = cl.order[k];
                        if (!hit[j] && dist(sp + 3 * s, p + 3 * j) <= seed_radius) hit[j] = 1;
                    }
                });
            for (int64_t i = 0; i < N; ++i) {
                if (!hit[i]) continue;
                int64_t c_i = 0;
                cl.for_cells_near(p + 3 * i, [&](int64_t cell) {
                    for (int64_t k = cl.start[cell]; k < cl.start[cell + 1]; ++k) {
                        const int64_t j = cl.order[k];
                        if (j == i) continue;
                        if (bipartite && !(q[i] * q[j] < 0.0 || q[i] == 0.0)) continue;
                        const double d = dist(p + 3 * i, p + 3 * j);
                        const double c = cut[int64_t(t[i]) * T + t[j]];
                        if (std::fabs(d - c) <= eps) {
                            undecided.push_back(i);
                            undecided.push_back(j);
                        } else if (d < c) {
                            ++c_i;
                        }
                    }
                });
                atoms.push_back(i);
                counts.push_back(c_i);
            }
        }
    }
    const size_t A = atoms.size(), M = undecided.size() / 2;
    return nb::make_tuple(to_numpy(std::move(atoms), {A}), to_numpy(std::move(counts), {A}),
                          to_numpy(std::move(undecided), {M, 2}));
}

// For every query point a, the number of reference points b with
// d(a, b) <= cutoff[qtype a, rtype b]; pairs with |d - cutoff| <= eps are
// returned undecided (as (a, b)) instead of counted.
static nb::tuple cross_counts(Points qpts, IntVec qtype, Points rpts, IntVec rtype, RealMat cutoff,
                              double eps) {
    const int64_t NQ = int64_t(qpts.shape(0)), NR = int64_t(rpts.shape(0));
    if (int64_t(qtype.shape(0)) != NQ || int64_t(rtype.shape(0)) != NR)
        throw nb::value_error("points and types must have the same length");
    const int64_t TQ = int64_t(cutoff.shape(0)), TR = int64_t(cutoff.shape(1));
    const double *qp = qpts.data(), *rp = rpts.data(), *cut = cutoff.data();
    const int32_t *qt = qtype.data(), *rt = rtype.data();
    check_types(qt, NQ, TQ);
    check_types(rt, NR, TR);

    std::vector<int64_t> counts(NQ, 0), undecided;
    {
        nb::gil_scoped_release release;
        if (NQ > 0 && NR > 0) {
            double rmax = 0.0;
            for (int64_t k = 0; k < TQ * TR; ++k) rmax = std::max(rmax, cut[k]);
            const CellList cl(rp, NR, rmax + eps);
            for (int64_t a = 0; a < NQ; ++a)
                cl.for_cells_near(qp + 3 * a, [&](int64_t cell) {
                    for (int64_t k = cl.start[cell]; k < cl.start[cell + 1]; ++k) {
                        const int64_t b = cl.order[k];
                        const double d = dist(qp + 3 * a, rp + 3 * b);
                        const double c = cut[int64_t(qt[a]) * TR + rt[b]];
                        if (std::fabs(d - c) <= eps) {
                            undecided.push_back(a);
                            undecided.push_back(b);
                        } else if (d < c) {
                            ++counts[a];
                        }
                    }
                });
        }
    }
    const size_t M = undecided.size() / 2;
    return nb::make_tuple(to_numpy(std::move(counts), {size_t(NQ)}),
                          to_numpy(std::move(undecided), {M, 2}));
}

NB_MODULE(_core, m) {
    m.doc() = "C++ kernels for nanocrystal-builder";
    m.def("pair_counts", &pair_counts, nb::arg("pts"), nb::arg("type_id"), nb::arg("charge"),
          nb::arg("cutoff"), nb::arg("bipartite"), nb::arg("eps"),
          "Coordination counts with per-type-pair cutoffs; returns (counts, undecided (M,2)).");
    m.def("pair_counts_local", &pair_counts_local, nb::arg("pts"), nb::arg("type_id"), nb::arg("charge"),
          nb::arg("cutoff"), nb::arg("bipartite"), nb::arg("eps"), nb::arg("seeds"), nb::arg("seed_radius"),
          "pair_counts for the atoms within seed_radius of any seed; returns (atoms, counts, undecided).");
    m.def("cross_counts", &cross_counts, nb::arg("qpts"), nb::arg("qtype"), nb::arg("rpts"), nb::arg("rtype"),
          nb::arg("cutoff"), nb::arg("eps"),
          "Reference points within cutoff[qtype, rtype] of each query; returns (counts, undecided (M,2)).");
}
