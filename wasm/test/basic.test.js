/**
 * Smoke tests for the libqe WASM module.
 * Run with: npm test  (requires dist/ to exist — run `npm run build` first)
 */
import loadLibQE from '../js/index.js';

let qe;
beforeAll(async () => { qe = await loadLibQE(); });

// ── Adjacency ─────────────────────────────────────────────────────────────────

test('connection_names returns correct pairs', () => {
    const pairs = qe.connection_names(['A', 'B', 'C']);
    expect(pairs).toEqual(['A & B', 'A & C', 'B & C']);
});

test('connection_indices has correct shape for n=3', () => {
    const idx = qe.connection_indices(3);
    expect(idx.rows.length).toBe(3);   // choose_two(3) = 3
    expect(idx.cols.length).toBe(3);
});

// ── Normalization ─────────────────────────────────────────────────────────────

test('normalize_networks: each row has unit L2 norm', () => {
    // 3×3 identity, row-major
    const data = new Float64Array([1,0,0, 0,1,0, 0,0,1]);
    const out  = qe.normalize_networks(data, 3, 3);
    for (let r = 0; r < 3; r++) {
        let norm = 0;
        for (let c = 0; c < 3; c++) norm += out.data[r * 3 + c] ** 2;
        expect(Math.sqrt(norm)).toBeCloseTo(1.0, 10);
    }
});

// ── Modeling ─────────────────────────────────────────────────────────────────

test('center_points: column means are zero', () => {
    const data = new Float64Array([1,2, 3,4, 5,6]);  // 3×2 row-major
    const out  = qe.center_points(data, 3, 2);
    // col 0: mean=3 → [1-3, 3-3, 5-3] = [-2, 0, 2]
    expect(out.data[0]).toBeCloseTo(-2, 10);
    expect(out.data[2]).toBeCloseTo( 0, 10);
    expect(out.data[4]).toBeCloseTo( 2, 10);
});

// ── Door ─────────────────────────────────────────────────────────────────────

test('door_lookback_block: omitted segment_ids defaults to no segments', () => {
    const data = new Float64Array([1, 0, 2, 1, 3, 0]);
    const out = qe.door_lookback_block(data, 3, 2, 2, false, false);
    expect(Array.from(out.data)).toEqual([1, 0, 3, 1, 5, 1]);
});

test('door blocks match ETM NA handling', () => {
    const lookbackData = new Float64Array([1, 1, NaN, 3, 5, NaN]);
    const lookback = qe.door_lookback_block(lookbackData, 3, 2, 3, true, false);
    expect(Array.from(lookback.data)).toEqual([1, 1, 1, 2, 3, 2]);

    const emaData = new Float64Array([10, 0, NaN, 10, 0, NaN]);
    const ema = qe.door_ema_block(emaData, 3, 2, 0.5);
    expect(Array.from(ema.data)).toEqual([10, 0, 10, 5, 5, 5]);
});

// ── Trajectory ───────────────────────────────────────────────────────────────

test('fit_trajectory_poly: JS wrapper supplies orthogonal basis default', () => {
    const data = new Float64Array([
        0, 0,
        1, 2,
        2, 4,
        3, 6,
    ]);
    const fit = qe.fit_trajectory_poly(data, 4, 2, [], 3, 1, 'loocv');
    expect(fit.degree).toBe(1);
    expect(fit.basis).toBe('orthogonal');

    const evaluated = qe.eval_trajectory_curve(fit.coeffs_x, fit.coeffs_y, [0, 1]);
    expect(evaluated.rows).toBe(2);
    expect(evaluated.cols).toBe(2);
    expect(evaluated.data[0]).toBeCloseTo(0, 10);
    expect(evaluated.data[1]).toBeCloseTo(0, 10);
    expect(evaluated.data[2]).toBeCloseTo(3, 10);
    expect(evaluated.data[3]).toBeCloseTo(6, 10);
});

// ── Trajectory without LAPACK ─────────────────────────────────────────────────
// This build has no LAPACK: the Gauss-Legendre rule and the least-squares
// inverse go through libqe's linalg fallbacks (they threw "use of LAPACK must
// be enabled" before).  Expected values are libqe's R build (LAPACK).

const CA = [[0, 1, 0.5], [0, 0.5, 0.2]];
const CB = [[0.1, 0.9, 0.4], [0, 0.4, 0.3]];

test('integrated_trajectory_distance matches the LAPACK build', () => {
    expect(qe.integrated_trajectory_distance(CA[0], CA[1], CB[0], CB[1], 0, 1))
        .toBeCloseTo(0.0584806531795148, 12);
});

test('lagged_trajectory_distance matches the LAPACK build', () => {
    expect(qe.lagged_trajectory_distance(CA[0], CA[1], CB[0], CB[1], 0.1))
        .toBeCloseTo(0.15513691840393, 12);
});

test('fit_trajectory_poly: cubic and LOOCV degree match the LAPACK build', () => {
    // 20 points; a cubic needs a 4×4 inverse (only 2×2 worked without LAPACK).
    const n = 20, pts = new Float64Array(2 * n), t = [];
    for (let i = 0; i < n; i++) {
        const s = i / (n - 1);
        t.push(s);
        pts[2 * i]     = Math.sin(3 * s) + 0.01 * ((i * 7) % 5);
        pts[2 * i + 1] = Math.cos(2 * s) + 0.01 * ((i * 3) % 4);
    }
    const fit = qe.fit_trajectory_poly(pts, n, 2, t, 3, 3, 'loocv', 'orthogonal');
    expect(fit.degree).toBe(3);
    const want_x = [-0.0186102489407905, 3.83212702497795, -3.51864879577555, -0.157025311602699];
    const want_y = [1.00837644921008, 0.134759749353767, -2.63244076984793, 1.08653110032508];
    want_x.forEach((v, i) => expect(fit.coeffs_x[i]).toBeCloseTo(v, 10));
    want_y.forEach((v, i) => expect(fit.coeffs_y[i]).toBeCloseTo(v, 10));

    expect(qe.fit_trajectory_poly(pts, n, 2, t, 5, 0, 'loocv', 'orthogonal').degree).toBe(4);
});

test('diagonal layout (optional trailing argument)', () => {
    expect(qe.n_connections(3)).toBe(3);
    expect(qe.n_connections(3, true)).toBe(6);
    expect(qe.connection_names(['A', 'B', 'C'], true))
        .toEqual(['A & A', 'A & B', 'B & B', 'A & C', 'B & C', 'C & C']);
    expect(qe.connection_names(['A', 'B', 'C'])).toEqual(['A & B', 'A & C', 'B & C']);
    const idx = qe.connection_indices(3, true);
    expect(Array.from(idx.rows)).toEqual([0, 0, 1, 0, 1, 2]);
    expect(Array.from(idx.cols)).toEqual([0, 1, 1, 2, 2, 2]);
    const v = [1, 2, 3, 4, 5, 6, 7, 8, 9];          // directed 3x3, column-major
    expect(Array.from(qe.fold_directed_network(v, true))).toEqual([1, 6, 5, 10, 14, 9]);
    expect(Array.from(qe.fold_directed_network(v))).toEqual([6, 10, 14]);
    // network_to_vector takes row-major data: row-major of the same matrix
    const rowMajor = [1, 4, 7, 2, 5, 8, 3, 6, 9];
    expect(Array.from(qe.network_to_vector(rowMajor, 3, 3, false, true))).toEqual([1, 4, 5, 7, 8, 9]);
});
