# Relative-stratified Geometry Pair sampling plan

## Objective

Replace manifest-neighbor pairing with deterministic, per-epoch, same-module
Geometry Pair sampling for known-module, new-stimulus training. Endpoint-local
graph and density-volume supervision must be sample-balanced; pair-local IoU
supervision must remain uniform over unique pairs.

## Fixed decisions

- Each eligible sample acts as an anchor once per training epoch.
- Each anchor draws at most 128 same-module candidates without replacement.
- Candidates are ranked per anchor by Coverage Similarity and split into disjoint
  anchor-local quartile regions.
- Each anchor requests two relative-low, one relative-mid, and one relative-high
  partner.
- Training uses deterministic `sampling_seed + epoch` resampling. Validation uses
  a fixed seed and fixed pair set.
- An unordered pair may occur at most once per epoch. Anchors are processed in a
  seeded random order; duplicate conflicts are repaired within the requested
  stratum. An exhausted stratum may leave its quota unfilled.
- Samples with sampled degree zero do not participate in joint training.
- Graph and density-volume endpoint losses use inverse-degree weights while
  preserving equal weighting across Coverage Types. IoU loss is uniform over
  unique Geometry Pairs.
- Train/validation splitting keeps `(dataset_dir, test_id)` groups intact and
  ensures every validation module is represented in training.
- Existing aggregate validation metrics remain; per-stratum model metrics are
  out of scope.
- New sampler options replace `contrastive_pairs_per_sample` without a
  compatibility alias.

## Sampling algorithm

1. Group eligible sample indices by `(dataset_dir, module_name)`. Preserve the
   existing requirement that both endpoints use the same coverage element
   universes.
2. Derive a stable RNG seed from the configured sampling seed, epoch, dataset
   directory, and module name. Do not use Python's process-randomized `hash()`.
3. Put every sample in the module group into a seeded random anchor order.
4. For each anchor, sort the other sample IDs into a stable order, select at most
   128 candidates without replacement, and compute their typed geometry targets
   and scalar Coverage Similarity.
5. Sort candidates by `(Coverage Similarity, seeded tie breaker)` and partition
   ranks into disjoint low quartile, middle half, and high quartile regions.
   For very small pools, create the largest disjoint regions possible; never
   duplicate a partner merely to fill a quota. Equal similarity values are
   separated only by the seeded tie breaker, so the strata remain explicitly
   relative rather than claiming an absolute semantic difference.
6. Draw the configured `2/1/1` quota from the three regions. Reject an unordered
   pair already selected by an earlier anchor and try another candidate in the
   same region. Leave the slot empty if that region is exhausted.
7. Compute each sample's final undirected degree from the deduplicated pair list.
   For every positive-degree endpoint use
   `endpoint_weight = mean_positive_degree / sample_degree`.
8. Materialize pair records with both indices, typed geometry targets, endpoint
   degrees, and endpoint weights. Log pair-set diagnostics and a stable pair-set
   fingerprint.

## Data split

Add a deterministic grouped split over `(dataset_dir, test_id)` rather than
individual sample indices. Greedily select validation stimulus groups toward the
configured validation fraction, but never remove the last training occurrence
of a module. Keep modules with insufficient distinct Test Stimuli in training.
Fail with a clear message if validation is required but no eligible grouped split
can be formed.

The split must satisfy:

- train and validation Test Stimulus keys are disjoint;
- every module represented in validation is represented in training;
- every sample belonging to a Test Stimulus stays on one side of the split;
- pair endpoints remain within their assigned split.

## Loss weighting

Add endpoint weights to `ContrastivePairBatch`. For graph loss, concatenate the
two endpoint predictions, targets, masks, and weights. For each Coverage Type,
compute a weighted mean over valid endpoint targets, then average the defined
type losses. This preserves the current type-equal objective.

Apply the same per-type weighted reduction to the two endpoint volume errors.
Do not apply endpoint weights to IoU errors: every unique Geometry Pair has one
equal IoU contribution. Ordinary non-joint graph training keeps its existing
unweighted reduction.

## Epoch and distributed behavior

Expose `resample(epoch)` on the training pair dataset. The joint DataModule must
resample before constructing each epoch's training DataLoader. Configure
Lightning to reload the training DataLoader every epoch so a changed pair count
cannot leave a stale `DistributedSampler` length or stale worker dataset copy.

All ranks build the same pair list from the same stable seed; Lightning's
distributed sampler partitions that list. Do not include rank in pair generation.
On checkpoint resume, `current_epoch` must reproduce the pair set for that epoch.
The validation pair dataset is built once with its fixed seed and is not
resampled.

## Configuration changes

Replace `contrastive_pairs_per_sample` and its CLI option with:

- `pair_candidate_pool_size` / `--pair-candidate-pool-size`, default `128`;
- `pair_relative_low_quota` / `--pair-relative-low-quota`, default `2`;
- `pair_relative_mid_quota` / `--pair-relative-mid-quota`, default `1`;
- `pair_relative_high_quota` / `--pair-relative-high-quota`, default `1`;
- `pair_sampling_seed` / `--pair-sampling-seed`, default `42`.

Reject negative quotas, a non-positive candidate pool, and a total quota larger
than the candidate pool. Update training scripts and README examples. Old training
commands intentionally fail on the removed option; inference checkpoint loading
is unaffected because model geometry is unchanged.

## File-level work

1. `datasets/datamodule.py` and a focused split helper: implement Test Stimulus
   grouped splitting and expose split diagnostics.
2. `datasets/pair_datamodule.py`: introduce a pure relative-stratified pair
   builder, pair records, epoch resampling, degree calculation, diagnostics, and
   endpoint-weight collation.
3. `datasets/data_types.py`: add endpoint degree/weight tensors to
   `ContrastivePairBatch`.
4. `models/losses.py`: add the type-equal weighted graph reduction used by joint
   batches without changing ordinary training behavior.
5. `models/contrastive_loss.py`: accept endpoint weights for volume reduction;
   retain uniform IoU reduction.
6. `trainer/joint_steps.py`: replace the two separately reduced graph losses with
   one weighted endpoint reduction and pass endpoint weights to volume loss.
7. `trainer/datamodule_factory.py` and `trainer/lightning_trainer.py`: pass sampler
   configuration, resample by current epoch, and reload the training DataLoader
   every joint-training epoch.
8. `trainer/config.py`, `main.py`, `run_contrastive.sh`, and `README.md`: replace
   the old option and document the relative policy.
9. Tests: replace manifest-neighbor assumptions and add sampler, split, weighting,
   epoch-resampling, and integration coverage.

## Required tests

- Two modules with very different absolute similarity ranges both receive the
  requested relative strata when candidates permit it.
- Reordering manifest records does not change sample-ID pair sets for the same
  seed and epoch.
- Same seed and epoch produce the same pair fingerprint; another epoch changes
  the training fingerprint; validation remains fixed.
- Candidate pools never exceed 128 and contain no repeated partner.
- No unordered pair is repeated; small groups underfill rather than duplicate.
- Every positive-degree sample is considered as an anchor once; degree-zero
  samples are absent from joint batches.
- Summed endpoint weight per positive-degree sample is equal, within tolerance,
  across a complete epoch.
- Graph and volume weighted reductions remain equal across Coverage Types with
  missing target masks; IoU reduction is unchanged and pair-uniform.
- Train and validation stimulus keys are disjoint, validation modules are a
  subset of training modules, and pair endpoints do not cross the split.
- A two-epoch Lightning smoke test observes different training pair fingerprints
  without stale sampler lengths, including a simulated distributed partition.
- Existing non-joint training, inference contracts, and typed geometry tests
  continue to pass.

## Runtime diagnostics

Log once per built pair set: eligible anchors, positive-degree samples,
degree-zero samples, unique pairs, requested and fulfilled quota by relative
stratum, duplicate conflicts, candidate-pool size statistics, degree min/mean/max,
absolute Coverage Similarity summary, and pair-set fingerprint. These are sampler
diagnostics, not additional per-stratum validation metrics.

## Acceptance criteria

The change is complete when all required tests pass, a two-epoch joint-training
smoke run deterministically changes only the training pair fingerprint, no
unordered pair duplicates exist, endpoint-local loss contribution is
degree-invariant over an epoch, and validation contains only unseen Test Stimuli
for modules present in training.
