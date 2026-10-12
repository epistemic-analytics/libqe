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
