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
