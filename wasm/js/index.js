/**
 * libqe WebAssembly — JavaScript entry point.
 *
 * Usage (Node.js or bundler):
 *
 *   import loadLibQE from '@qe-libs/libqe-wasm';
 *   const qe = await loadLibQE();
 *
 *   // Adjacency
 *   const pairs = qe.connection_names(['A', 'B', 'C']);
 *   // → ['A & B', 'A & C', 'B & C']
 *
 *   // Normalization
 *   const lw  = new Float64Array([1,1,0, 1,0,1]);
 *   const out = qe.normalize_networks(lw, 2, 3);
 *   // out → { data: Float64Array, rows: 2, cols: 3 }
 *
 * The ENA model code (rotations, node positions, CCD) is in
 * @qe-libs/rena-wasm and all accumulation in @qe-libs/tma-wasm.
 *
 * Matrix convention
 * -----------------
 * All matrix inputs are flat Float64Array in *row-major* order, paired with
 * explicit `rows` and `cols` arguments.  Return values are objects:
 *   { data: Float64Array, rows: number, cols: number }
 *
 * Helper to read a cell: data[row * cols + col]
 */

// Wrap the Emscripten factory so JS callers get stable defaults even though
// embind does not preserve C++ default arguments.
import createLibQE from '../dist/libqe.js';

export default async function loadLibQE(...args) {
  const qe = await createLibQE(...args);

  if (typeof qe.door_lookback_block === 'function') {
    const rawDoorLookbackBlock = qe.door_lookback_block.bind(qe);
    qe.door_lookback_block = (
      data,
      rows,
      cols,
      lookbackSize = 20,
      aggregateMean = false,
      weightingLinear = false,
      segmentIds = [],
    ) => rawDoorLookbackBlock(
      data,
      rows,
      cols,
      lookbackSize,
      aggregateMean,
      weightingLinear,
      segmentIds,
    );
  }

  if (typeof qe.door_ema_block === 'function') {
    const rawDoorEmaBlock = qe.door_ema_block.bind(qe);
    qe.door_ema_block = (
      data,
      rows,
      cols,
      alpha = 0.1,
      segmentIds = [],
    ) => rawDoorEmaBlock(data, rows, cols, alpha, segmentIds);
  }

  if (typeof qe.fit_trajectory_poly === 'function') {
    const rawFitTrajectoryPoly = qe.fit_trajectory_poly.bind(qe);
    qe.fit_trajectory_poly = (
      data,
      rows,
      cols,
      t = [],
      maxDegree = 3,
      fixedDegree = 0,
      criterion = 'loocv',
      basis = 'orthogonal',
    ) => rawFitTrajectoryPoly(
      data,
      rows,
      cols,
      t,
      maxDegree,
      fixedDegree,
      criterion,
      basis,
    );
  }

  return qe;
}
