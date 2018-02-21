//! Linear basis projection module.

use geometry::{Space, Span, Vector};


pub(crate) type ActivationT = f64;
pub(crate) type IndexT = usize;

pub(crate) type DenseT = Vector<ActivationT>;
pub(crate) type SparseT = Vector<IndexT>;


#[inline]
pub(self) fn l1(x: &[ActivationT]) -> ActivationT {
    x.into_iter().fold(0.0, |acc, v| acc + v.abs())
}


/// Projected feature vector representation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Projection {
    /// Dense, floating-point activation vector.
    Dense(DenseT),

    /// Sparse, index-based activation vector.
    Sparse(SparseT),
}

impl Projection {
    /// Compute the l1 normalisation constant of the projection.
    pub fn z(&self) -> ActivationT {
        match self {
            &Projection::Dense(ref activations) => l1(activations.as_slice().unwrap()),
            &Projection::Sparse(ref active_indices) => active_indices.len() as ActivationT,
        }
    }

    /// Return the maximum number of active features in the basis space.
    pub fn activity(&self) -> usize {
        match self {
            &Projection::Dense(ref activations) => activations.len(),
            &Projection::Sparse(ref active_indices) => active_indices.len(),
        }
    }

    /// Expand and normalise a given projection, and convert into a raw, dense
    /// vector.
    fn expanded(self, span: Span) -> DenseT {
        #[inline]
        fn expand_sparse(active_indices: SparseT, z: ActivationT, size: usize) -> DenseT {
            let mut phi = Vector::zeros((size,));

            let activation = 1.0 / z;
            for idx in active_indices.iter() {
                phi[*idx] = activation;
            }

            phi
        }

        match self.z() {
            z if z.abs() < 1e-6 => match self {
                Projection::Dense(phi) => phi,
                Projection::Sparse(active_indices) => {
                    expand_sparse(active_indices, 1.0, span.into())
                },
            },
            z => match self {
                Projection::Dense(phi) => phi.iter().map(|x| x / z).collect(),
                Projection::Sparse(active_indices) => {
                    expand_sparse(active_indices, z, span.into())
                },
            },
        }
    }
}

impl Into<Projection> for DenseT {
    fn into(self) -> Projection { Projection::Dense(self) }
}

impl Into<Projection> for SparseT {
    fn into(self) -> Projection { Projection::Sparse(self) }
}


/// Trait for basis projectors.
pub trait Projector<I: ?Sized>: Space<Repr = Projection> {
    /// Project data from an input space onto the basis.
    fn project(&self, input: &I) -> Projection;

    /// Project data from an input space onto the basis and convert into a raw,
    /// dense vector.
    fn project_expanded(&self, input: &I) -> DenseT { self.project(input).expanded(self.span()) }
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

impl_fixed!(1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24);


mod rbf_network;
pub use self::rbf_network::*;

mod fourier;
pub use self::fourier::*;

mod polynomial;
pub use self::polynomial::*;

mod tile_coding;
pub use self::tile_coding::*;

mod uniform_grid;
pub use self::uniform_grid::*;
