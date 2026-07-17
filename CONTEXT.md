# RTL Coverage Space

RTLMap learns coverage predictions and a geometric representation of the coverage space produced by a test stimulus for an RTL module.

## Language

**Coverage Type**:
One of line, condition, toggle, FSM, or branch coverage. Each type has its own element universe and geometric subspace.

**Coverage Set**:
The covered elements of one coverage type for a single sample.
_Avoid_: Coverage vector, when referring only to the covered elements rather than their serialized representation

**Test Stimulus**:
A dataset-local test input identified by `test_id`. Samples in the same dataset that share a Test Stimulus may describe different RTL modules exercised by the same input.
_Avoid_: Sample, when referring to the input shared across module-specific records

**Coverage Density**:
The cardinality of a Coverage Set divided by the number of elements in that Coverage Type's universe.
_Avoid_: Coverage volume

**Coverage Space**:
The five typed geometric subspaces that represent a sample's Coverage Sets.

**Coverage Subrectangle**:
A configured-dimensional axis-aligned rectangle for one Coverage Type whose true geometric volume represents Coverage Density.
_Avoid_: Embedding box, latent box

**Coverage Similarity**:
The equal-weighted mean of defined per-type Jaccard similarities. A type whose union is empty does not contribute.

**Geometric Similarity**:
The equal-weighted mean of defined per-type true-volume intersection-over-union values between Coverage Subrectangles.
_Avoid_: Mean axis IoU

**Geometry Pair**:
Two samples for the same RTL module whose typed Coverage Sets provide density and Jaccard supervision for Coverage Subrectangles.
_Avoid_: Cross-module pair

**Geometry Pair Candidate**:
A same-module sample considered as a possible partner for an anchor before uniqueness and stratum quotas select a Geometry Pair.
_Avoid_: Negative sample, candidate pair

**Geometry Pair Set**:
The unique Geometry Pairs selected for one dataset split and sampling epoch.
_Avoid_: Pair cache, pair batch

**Relative Similarity Stratum**:
The relative-low, relative-mid, or relative-high rank region of an anchor sample's same-module Geometry Pair candidates. A relative-low candidate is not necessarily low in absolute Coverage Similarity.
_Avoid_: Dissimilar pair, negative pair
