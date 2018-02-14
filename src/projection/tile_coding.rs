use super::{Projection, Projector};
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

    fn dim(&self) -> usize { unimplemented!() }

    fn size(&self) -> usize { self.memory_size as usize }

    fn activity(&self) -> usize { self.n_tilings }

    fn equivalent(&self, other: &Self) -> bool {
        self.size() == other.size() && self.n_tilings == other.n_tilings
            && self.memory_size == other.memory_size
    }
}
