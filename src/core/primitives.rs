use geometry::Vector;
use std::collections::BTreeSet;

pub type ActivationT = f64;
pub type IndexT = usize;

pub type IndexSet = BTreeSet<IndexT>;

pub type DenseT = Vector<ActivationT>;
pub type SparseT = IndexSet;
