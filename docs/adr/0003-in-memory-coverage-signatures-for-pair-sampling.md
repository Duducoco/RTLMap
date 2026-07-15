# Use in-memory Coverage Signatures for Geometry Pair sampling

The Geometry Pair sampler will parse and validate each sample's typed Coverage
Sets once into an in-memory Coverage Signature, score the 128 candidates from
that representation, and materialize full geometry targets only for selected
pairs. This preserves deterministic seed-plus-epoch sampling, per-epoch
resampling, the 2/1/1 relative quota policy, and cross-rank consistency while
avoiding a persistent similarity index and the repeated sparse-vector parsing
that currently dominates runtime. Concrete pair fingerprints may change from
the pre-optimization implementation, but per-stratum sampling statistics must
remain within the agreed tolerance.
