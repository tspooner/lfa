use crate::{
    basis::Composable,
    core::{Projection, Projector},
    geometry::{Card, Space, Vector},
};
use std::hash::{BuildHasher, Hasher};

#[inline]
fn bin_state(input: &[f64], n_tilings: usize) -> Vec<isize> {
    input
        .into_iter()
        .map(|f| (*f * n_tilings as f64).floor() as isize)
        .collect()
}

#[inline]
fn hash_state<H: Hasher>(
    mut hasher: H,
    state: &[isize],
    n_tilings: usize,
    memory_size: usize,
) -> Vec<usize>
{
    (0..n_tilings).map(|t| {
        let t = t as isize;
        let tx2 = t * 2;

        hasher.write_isize(t);

        for (i, s) in state.iter().enumerate() {
            let offset = t + i as isize * tx2;

            hasher.write_isize((s + offset) / n_tilings as isize)
        }

        hasher.finish() as usize % memory_size
    })
    .collect()
}

/// Generalised tile coding scheme with hashing.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct TileCoding<H> {
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

impl<H> Space for TileCoding<H> {
    type Value = Projection;

    fn dim(&self) -> usize { self.memory_size }

    fn card(&self) -> Card { unimplemented!() }
}

impl<H: BuildHasher> Projector<[f64]> for TileCoding<H> {
    fn project(&self, input: &[f64]) -> Projection {
        let state = bin_state(input, self.n_tilings);
        let hasher = self.hasher_builder.build_hasher();

        hash_state(hasher, &state, self.n_tilings, self.memory_size).into()
    }
}

impl<H: BuildHasher> Projector<Vec<f64>> for TileCoding<H> {
    fn project(&self, input: &Vec<f64>) -> Projection { Projector::<[f64]>::project(self, &input) }
}

impl<H: BuildHasher> Projector<Vector<f64>> for TileCoding<H> {
    fn project(&self, input: &Vector<f64>) -> Projection {
        Projector::<[f64]>::project(self, input.as_slice().unwrap())
    }
}

impl<H: BuildHasher> Composable for TileCoding<H> {}

#[cfg(test)]
mod tests {
    use super::*;
    use quickcheck::quickcheck;

    #[test]
    fn test_bin_state() {
        fn prop_output(state: f64, n_tilings: usize) -> bool {
            bin_state(&[state], n_tilings)[0] == (state * n_tilings as f64).floor() as isize
        }

        quickcheck(prop_output as fn(f64, usize) -> bool);

        assert_eq!(bin_state(&[0.0, 0.0], 16), vec![0, 0]);
        assert_eq!(bin_state(&[0.99, 0.99], 16), vec![15, 15]);
        assert_eq!(bin_state(&[1.0, 1.0], 16), vec![16, 16]);
        assert_eq!(bin_state(&[0.0, 1.0], 16), vec![0, 16]);
        assert_eq!(bin_state(&[-1.0, -1.0], 16), vec![-16, -16]);

        assert_eq!(bin_state(&[0.0, 0.5], 16), vec![0, 8]);
        assert_eq!(bin_state(&[0.5, 0.0], 16), vec![8, 0]);
        assert_eq!(bin_state(&[0.5, 0.5], 16), vec![8, 8]);
    }
}
