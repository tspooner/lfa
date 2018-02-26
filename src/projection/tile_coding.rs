use super::{Projection, Projector};
use geometry::{Space, Span};
use rand::ThreadRng;
use std::hash::{BuildHasher, Hasher};


#[inline]
fn bin_state(input: &[f64], n_tilings: usize) -> Vec<usize> {
    input.into_iter().map(|f| (*f*n_tilings as f64).floor() as usize).collect()
}


#[inline]
fn hash_state<H: Hasher>(mut hasher: H, state: &[usize],
                         n_tilings: usize, memory_size: usize) -> Vec<usize> {
    let state_len = state.len();

    (0..n_tilings).map(|t| {
        hasher.write_usize(t);
        for i in 0..state_len {
            hasher.write_usize((state[i] + t + i*t*2)/n_tilings)
        }

        hasher.finish() as usize % memory_size
    }).collect()
}


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
        let state = bin_state(input, self.n_tilings);
        let hasher = self.hasher_builder.build_hasher();

        hash_state(hasher, &state, self.n_tilings, self.memory_size).into()
    }
}
