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

// ── Accumulation ─────────────────────────────────────────────────────────────

test('accumulate_stanza: output shape matches input rows', () => {
    // 3 rows × 3 codes → choose_two(3)=3 connection columns
    const codes = new Float64Array([1,1,0, 1,0,1, 0,1,1]);
    const out   = qe.accumulate_stanza(codes, 3, 3, 2, 0, true);
    expect(out.rows).toBe(3);
    expect(out.cols).toBe(3);
});

test('rolling_window_sum: single row window is identity', () => {
    const codes = new Float64Array([1,2,3, 4,5,6, 7,8,9]);
    const out   = qe.rolling_window_sum(codes, 3, 3, 1);
    expect(Array.from(out.data)).toEqual([1,2,3, 4,5,6, 7,8,9]);
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

test('flat_index matches column-major expectation', () => {
    // 3D array of dims [2,3,4], index [1,2,3] → 1 + 2*2 + 3*6 = 23
    expect(qe.flat_index([1,2,3], [2,3,4])).toBe(23);
});

// ── Modeling — node_positions regression ──────────────────────────────────────

test('node_positions: nodes.cols equals num_dims when num_dims passed explicitly', () => {
    // 3 units × 3 connections (choose(3,2)), 2 dimensions
    // adj_mats: 3×3 row-major
    const adj  = new Float64Array([1,0,0, 0,1,0, 0,0,1]);
    // t (unit points): 3×2 row-major
    const t    = new Float64Array([0.1,0.2, 0.3,0.4, 0.5,0.6]);
    const r    = qe.node_positions(adj, 3, 3, t, 3, 2, 2);
    expect(r.nodes.rows).toBe(3);   // 3 codes → 3 nodes
    expect(r.nodes.cols).toBe(2);   // 2 dimensions
    expect(r.nodes.data.length).toBe(6);
});

test('node_positions: nodes.cols is non-zero when num_dims omitted (defaults to t_cols)', () => {
    // Regression: Embind passes 0 for missing int args; omitting num_dims used to
    // produce { rows: 3, cols: 0 } because ssX was created as 0×num_nodes.
    const adj  = new Float64Array([1,0,0, 0,1,0, 0,0,1]);
    const t    = new Float64Array([0.1,0.2, 0.3,0.4, 0.5,0.6]);
    // Simulate the caller omitting num_dims (Embind receives 0)
    const r    = qe.node_positions(adj, 3, 3, t, 3, 2, 0);
    expect(r.nodes.cols).toBeGreaterThan(0);
    expect(r.nodes.cols).toBe(2);
});

// ── Accumulation — extended ───────────────────────────────────────────────────

test('accumulate_stanza: ordered=true gives n_codes² columns', () => {
    // 3 rows × 3 codes, ordered → 3² = 9 cols
    const codes = new Float64Array([1,1,0, 1,0,1, 0,1,1]);
    const out   = qe.accumulate_stanza(codes, 3, 3, 2, 0, true, true);
    expect(out.rows).toBe(3);
    expect(out.cols).toBe(9);
});

test('connection_matrix: ordered=false result is symmetric', () => {
    const g = new Float64Array([1, 0, 1]);
    const r = new Float64Array([1, 1, 0]);
    const m = qe.connection_matrix(g, 3, r, 3, 1.0, false);
    // unordered result is symmetric: m[i,j] == m[j,i] (row-major, 3×3)
    expect(m.data[1]).toBeCloseTo(m.data[3], 10);   // (0,1) vs (1,0)
    expect(m.data[2]).toBeCloseTo(m.data[6], 10);   // (0,2) vs (2,0)
    expect(m.data[5]).toBeCloseTo(m.data[7], 10);   // (1,2) vs (2,1)
});

test('accumulate_unit: output length is choose_two(n_codes) when unordered', () => {
    // 4 rows × 3 codes; unit covers rows [0, 1, 2]; decay = constant 1
    const codes    = new Float64Array([1,0,0, 0,1,0, 0,0,1, 1,1,0]);
    const unitRows = new Int32Array([0, 1, 2]);
    const decayFn  = (d) => new Float64Array(d.length).fill(1.0);
    const out = qe.accumulate_unit(codes, 4, 3, unitRows, decayFn, false);
    // choose(3,2) = 3
    expect(out.length).toBe(3);
});

test('accumulate_unit: output length is n_codes² when ordered', () => {
    const codes    = new Float64Array([1,0,0, 0,1,0, 0,0,1, 1,1,0]);
    const unitRows = new Int32Array([0, 1, 2]);
    const decayFn  = (d) => new Float64Array(d.length).fill(1.0);
    const out = qe.accumulate_unit(codes, 4, 3, unitRows, decayFn, true);
    expect(out.length).toBe(9);
});

test('accumulate_unit_with_rows: networks + row_networks returned', () => {
    const codes    = new Float64Array([1,0, 0,1, 1,1]);
    const unitRows = new Int32Array([0, 1, 2]);
    const decayFn  = (d) => new Float64Array(d.length).fill(1.0);
    const out = qe.accumulate_unit_with_rows(codes, 3, 2, unitRows, decayFn, false);
    // choose(2,2) = 1 connection
    expect(out.networks.length).toBe(1);
    // row_networks: always 3 rows × n_codes² cols (full square, before folding)
    expect(out.row_networks.rows).toBe(3);
    expect(out.row_networks.cols).toBe(4);   // 2² = 4 for n_codes=2
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

// ── Rotation ──────────────────────────────────────────────────────────────────

test('ena_svd: returns rotation matrix and eigenvalues', () => {
    // 3 units × 2 connection dimensions
    const data = new Float64Array([1,0, 0,1, 1,1]);
    const r    = qe.ena_svd(data, 3, 2);
    expect(r.rotation.rows).toBe(2);
    expect(r.rotation.cols).toBe(2);
    expect(r.eigenvalues.length).toBe(2);
    expect(r.column_names.length).toBe(2);
});

test('ena_svd: does not throw for exact value 0.025 (regression: Jacobi NaN via 0/0)', () => {
    // A 2×3 rank-1 matrix whose Jacobi iteration converges to an all-zero
    // 2×2 sub-block (aqq = arr = aqr = 0 exactly).  The `<` skip condition
    // evaluated `0 < 0` as false, fell through to theta = 0/0 = NaN, and
    // terminated via std::terminate.  Fixed by using `<=`.
    const bad = new Float64Array([0.296, 0.025, -0.274, -0.296, -0.025, 0.274]);
    expect(() => qe.ena_svd(bad, 2, 3)).not.toThrow();
    const r = qe.ena_svd(bad, 2, 3);
    expect(r.eigenvalues.length).toBe(3);
    // One non-zero eigenvalue; the other two are zero (rank-1 input)
    const evSorted = [...r.eigenvalues].sort((a, b) => b - a);
    expect(evSorted[0]).toBeGreaterThan(0.1);
    expect(evSorted[1]).toBeCloseTo(0, 10);
    expect(evSorted[2]).toBeCloseTo(0, 10);
});

test('deflate: removes variance along axis', () => {
    // data aligned with first standard basis vector; deflating that axis
    // should zero out the first column
    const data = new Float64Array([1,0, 2,0, 3,0]);  // 3×2 row-major
    const axis = new Float64Array([1, 0]);
    const out  = qe.deflate(data, 3, 2, axis);
    expect(out.rows).toBe(3);
    expect(out.cols).toBe(2);
    // col 0 of output should be near zero
    [0, 2, 4].forEach(i => expect(Math.abs(out.data[i])).toBeCloseTo(0, 8));
});

test('accumulate_tensor_unit: default 1-D tensor smoke test', () => {
    // Default tensor: dims=[2], tensor=[weight=1, window=2].
    // IS_DEFAULT path: every ground row within 2 time-units gets weight=1.
    //
    // 4 context rows × 2 codes, times = [0,1,2,3], all rows belong to the unit.
    // No extra context factors — supply a 4×1 all-zeros context_lookup (values
    // are only read in the non-default path, so content doesn't matter here).

    const tensor  = new Float64Array([1, 2]);   // [weight=1, window=2]
    const dims    = new Int32Array([2]);         // 1-D tensor of size 2

    // No extra context factors — supply a single dummy column of zeros
    const cl      = new Int32Array([0, 0, 0, 0]);  // 4×1, all zeros
    const urows   = new Int32Array([0, 1, 2, 3]);
    const codes   = new Float64Array([1,0, 0,1, 1,1, 0,0]);
    const times   = new Float64Array([0, 1, 2, 3]);

    const out = qe.accumulate_tensor_unit(
        tensor, dims,
        new Int32Array([]),   // dims_sender  (none)
        new Int32Array([]),   // dims_receiver (none)
        new Int32Array([]),   // dims_mode    (none)
        cl, 4, 1,
        urows, codes, 4, 2, times,
        /*ordered=*/true
    );

    // connection_counts: length n_codes² = 4
    expect(out.connection_counts.length).toBe(4);
    // row_connection_counts: n_unit_rows × n_codes² = 4×4
    expect(out.row_connection_counts.rows).toBe(4);
    expect(out.row_connection_counts.cols).toBe(4);
});

test('means_rotation: returns rotation with column_names', () => {
    // 4 units × 4 ENA dims (row-major).  Group a = rows 0–1, group b = rows 2–3.
    const data = new Float64Array([
        1, 0, 0, 1,   // unit 0 — group a
        0, 1, 1, 0,   // unit 1 — group a
        2, 1, 1, 2,   // unit 2 — group b
        1, 2, 2, 1,   // unit 3 — group b
    ]);
    const groupPairs = [{ a: new Int32Array([0, 1]), b: new Int32Array([2, 3]) }];
    const r = qe.means_rotation(data, 4, 4, groupPairs);
    expect(r.rotation.rows).toBe(4);
    expect(r.rotation.cols).toBe(4);
    expect(r.eigenvalues.length).toBe(4);
    expect(r.column_names[0]).toBe('MR1');
});

test('generalized_means_rotation: numeric target returns GMR1/SVD2', () => {
    // 10 units × 3 ENA dims, numeric target, no covariates, no y axis.
    // Use diverse floating-point data to avoid numerical edge cases in the
    // no-BLAS/LAPACK WASM environment (e.g. near-zero after orthogonalization).
    const V = new Float64Array([
        0.5, 0.2, 0.8,
        0.3, 0.7, 0.1,
        0.8, 0.4, 0.6,
        0.1, 0.9, 0.3,
        0.6, 0.1, 0.7,
        0.4, 0.8, 0.2,
        0.7, 0.3, 0.9,
        0.2, 0.6, 0.4,
        0.9, 0.5, 0.1,
        0.3, 0.4, 0.7,
    ]);
    const xTarget = new Float64Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
    const xModel  = xTarget;                   // 10×1 model matrix
    const x1Cols  = new Int32Array([0]);        // only column is target
    const xSubset = new Int32Array([]);         // use all rows
    const dummy   = new Float64Array(10);       // ignored y params
    const dummy0  = new Int32Array([]);

    const r = qe.generalized_means_rotation(
        V,      10, 3,
        xModel, 10, 1, xTarget, x1Cols,
        /*x_categorical=*/false, /*x_n_groups=*/0, xSubset,
        /*has_y=*/false,
        dummy, 10, 1, dummy, dummy0,
        /*y_categorical=*/false, /*y_n_groups=*/0,
        /*n_lambda=*/10, /*k_folds=*/3, /*lasso_eps=*/0.01
    );
    expect(r.rotation.rows).toBe(3);
    expect(r.rotation.cols).toBe(3);
    expect(r.eigenvalues.length).toBe(3);
    expect(r.column_names[0]).toBe('GMR1');
    expect(r.column_names[1]).toBe('SVD2');
});

test('ccd_window: estimates a window and returns per-lag curves', () => {
    // Two conversations × 30 rows, 3 codes. Code B tends to follow code A at
    // lag 1 (with decay), so the corrected covariance peaks early and decays.
    let seed = 3;
    const rnd = () => (seed = (seed * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff;
    const flat = []; let nRows = 0;
    const groupSizes = [], rowIndices = [];
    for (let c = 0; c < 2; c++) {
        let prevA = 0;
        for (let i = 0; i < 30; i++) {
            const A = rnd() < 0.4 ? 1 : 0;
            const B = (prevA === 1 && rnd() < 0.8) ? 1 : (rnd() < 0.15 ? 1 : 0);
            const Cc = rnd() < 0.3 ? 1 : 0;
            flat.push(A, B, Cc);
            rowIndices.push(nRows++);
            prevA = A;
        }
        groupSizes.push(30);
    }
    const r = qe.ccd_window(flat, nRows, 3, groupSizes, rowIndices, 12, 5);

    expect(r.window_size).toBeGreaterThanOrEqual(1);
    expect(r.window_size).toBeLessThanOrEqual(12);
    expect(r.peak_lag).toBeGreaterThanOrEqual(1);
    // Curves are indexed by lag 0..max_window.
    expect(r.lag.length).toBe(13);
    expect(r.frob.length).toBe(13);
    expect(r.frob_unbiased_signed.length).toBe(13);
    expect(r.total_weight.length).toBe(13);
    expect(r.lag[0]).toBe(0);
    expect(r.lag[12]).toBe(12);
});

test('ccd_window: conversations shorter than min_overlap default to window 1', () => {
    const r = qe.ccd_window([1, 0, 1, 0, 1, 0], 2, 3, [2], [0, 1], 12, 5);
    expect(r.window_size).toBe(1);
    expect(r.peak_lag).toBe(0);
});
