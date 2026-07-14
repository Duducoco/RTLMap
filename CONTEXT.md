# RTL Coverage Space

RTLMap learns coverage predictions and a geometric representation of the coverage space produced by a test stimulus for an RTL module.

## Language

**Coverage Type**:
One of line, condition, toggle, FSM, or branch coverage. Each type has its own element universe and geometric subspace.

**Coverage Set**:
The covered elements of one coverage type for a single sample.
_Avoid_: Coverage vector, when referring only to the covered elements rather than their serialized representation

**Coverage Density**:
The cardinality of a Coverage Set divided by the number of elements in that Coverage Type's universe.
_Avoid_: Coverage volume

**Coverage Space**:
The five typed geometric subspaces that represent a sample's Coverage Sets.

**Coverage Subrectangle**:
A ten-dimensional axis-aligned rectangle for one Coverage Type whose true geometric volume represents Coverage Density.
_Avoid_: Embedding box, latent box

**Coverage Similarity**:
The equal-weighted mean of defined per-type Jaccard similarities. A type whose union is empty does not contribute.

**Geometric Similarity**:
The equal-weighted mean of defined per-type true-volume intersection-over-union values between Coverage Subrectangles.
_Avoid_: Mean axis IoU
