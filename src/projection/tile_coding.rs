use super::{Projection, Projector};
use geometry::{Space, Span};
use rand::ThreadRng;
use std::hash::{BuildHasher, Hasher};

/// Generalised tile coding scheme with hashing.
#[derive(Clone, Serialize, Deserialize)]
pub struct TileCoding<H: BuildHasher> {
    hasher_builder: H,
    n_tilings: usize,
    memory_size: usize,
}

impl<H: BuildHasher> TileCoding<H> {
    pub fn new(hasher_builder: H, n_tilings: usize, memory_size: usize) -> Self {
        TileCoding {
            hasher_builder: hasher_builder,
            n_tilings: n_tilings,
            memory_size: memory_size,
        }
    }
}

impl<H: BuildHasher> Space for TileCoding<H> {
    type Repr = Projection;

    fn sample(&self, _rng: &mut ThreadRng) -> Projection { unimplemented!() }

    fn dim(&self) -> usize { unimplemented!() }

    fn span(&self) -> Span { Span::Finite(self.memory_size) }
}

impl<H: BuildHasher> Projector<[f64]> for TileCoding<H> {
    fn project(&self, input: &[f64]) -> Projection {
        let mut hasher = self.hasher_builder.build_hasher();

        let n_floats = input.len();
        let floats: Vec<usize> = input
            .iter()
            .map(|f| (*f * self.n_tilings as f64).floor() as usize)
            .collect();

        Projection::Sparse(
            (0..self.n_tilings)
                .map(|t| {
                    hasher.write_usize(t);
                    for i in 0..n_floats {
                        hasher.write_usize((floats[i] + t + i * t * 2) / self.n_tilings)
                    }

                    hasher.finish() as usize % self.memory_size
                })
                .collect(),
        )
    }
}
