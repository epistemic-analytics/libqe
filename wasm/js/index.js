/**
 * libqe WebAssembly — JavaScript entry point.
 *
 * Usage (Node.js or bundler):
 *
 *   import loadLibQE from '@qe-libs/libqe-wasm';
 *   const qe = await loadLibQE();
 *
 *   // Adjacency
 *   const pairs = qe.svector_to_upper_tri(['A', 'B', 'C']);
 *   // → ['A & B', 'A & C', 'B & C']
 *
 *   // Accumulation
 *   const codes = new Float64Array([1,1,0, 1,0,1, 0,1,1]);
 *   const out   = qe.stanza_window(codes, 3, 3, 2, 0, true);
 *   // out → { data: Float64Array, rows: 3, cols: 3 }
 *
 * Matrix convention
 * -----------------
 * All matrix inputs are flat Float64Array in *row-major* order, paired with
 * explicit `rows` and `cols` arguments.  Return values are objects:
 *   { data: Float64Array, rows: number, cols: number }
 *
 * Helper to read a cell: data[row * cols + col]
 */

// Re-export the Emscripten factory function produced by the WASM build.
// In a bundler (webpack / vite) this import resolves to dist/libqe.js.
export { default } from '../dist/libqe.js';
