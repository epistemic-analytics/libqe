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
