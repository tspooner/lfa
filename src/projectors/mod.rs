use geometry::{Space, Vector};
use std::collections::{BTreeSet, HashMap};

pub type ActivationT = f64;
pub type IndexT = usize;

pub type IndexSet = BTreeSet<IndexT>;

pub type DenseT = Vector<ActivationT>;
pub type SparseT = IndexSet;

mod projection;
pub use self::projection::Projection;

mod feature;
pub use self::feature::{Feature, CandidateFeature};

pub mod adaptive;
pub mod fixed;

/// Trait for basis projectors.
pub trait Projector<I: ?Sized>: Space<Value = Projection> {
    /// Project data from an input space onto the basis.
    fn project(&self, input: &I) -> Projection;

    /// Project data from an input space onto the basis and convert into a raw,
    /// dense vector.
    fn project_expanded(&self, input: &I) -> DenseT { self.project(input).expanded(self.dim()) }
}

/// Trait for projectors with adaptive bases.
pub trait AdaptiveProjector<I: ?Sized>: Projector<I> {
    fn discover(&mut self, input: &I, error: f64) -> Option<HashMap<usize, IndexSet>>;
    fn add_feature(&mut self, feature: CandidateFeature) -> Option<(usize, IndexSet)>;
}

impl<P: Projector<[f64]>> Projector<Vec<f64>> for P {
    fn project(&self, input: &Vec<f64>) -> Projection { Projector::<[f64]>::project(self, &input) }
}

macro_rules! impl_fixed {
    ($($n:expr),*) => {
        $(
            impl<P: Projector<[f64]>> Projector<[f64; $n]> for P {
                fn project(&self, input: &[f64; $n]) -> Projection {
                    Projector::<[f64]>::project(self, input)
                }
            }
        )*
    }
}

impl_fixed!(
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    9,
    10,
    11,
    12,
    13,
    14,
    15,
    16,
    17,
    18,
    19,
    20,
    21,
    22,
    23,
    24
);
