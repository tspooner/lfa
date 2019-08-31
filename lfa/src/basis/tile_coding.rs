use crate::{IndexT, ActivationT, Features, Result, check_index, basis::Projector};
use std::hash::{BuildHasher, Hasher};

#[inline]
fn bin_state(input: &[f64], n_tilings: usize) -> Vec<isize> {
    input
        .into_iter()
        .map(|f| (*f * n_tilings as f64).floor() as isize)
        .collect()
}

#[inline]
fn hash_state<'a, H: Hasher + 'a>(
    mut hasher: H,
    state: &'a [isize],
    n_tilings: usize,
    memory_size: usize,
) -> impl Iterator<Item = usize> + 'a
{
    (0..n_tilings).map(move |t| {
        let t = t as isize;
        let tx2 = t * 2;

        hasher.write_isize(t);

        for (i, s) in state.iter().enumerate() {
            let offset = t + i as isize * tx2;

            hasher.write_isize((s + offset) / n_tilings as isize)
        }

        hasher.finish() as usize % memory_size
    })
}

/// Generalised tile coding scheme with hashing.
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
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

impl<H: BuildHasher> Projector for TileCoding<H> {
    fn n_features(&self) -> usize { self.memory_size }

    fn project_ith(&self, input: &[f64], index: IndexT) -> Result<Option<ActivationT>> {
        check_index(index, self.memory_size, || {
            let state = bin_state(input, self.n_tilings);
            let mut hash = hash_state(
                self.hasher_builder.build_hasher(),
                &state,
                self.n_tilings,
                self.memory_size
            );

            Ok(if hash.any(|f| index == f) { Some(1.0) } else { None })
        })
    }

    fn project(&self, input: &[f64]) -> Result<Features> {
        let state = bin_state(input, self.n_tilings);
        let hash = hash_state(
            self.hasher_builder.build_hasher(),
            &state,
            self.n_tilings,
            self.memory_size
        );

        Ok(Features::Sparse(self.memory_size, hash.map(|i| (i, 1.0)).collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    quickcheck! {
        fn test_bin_state_1d(state: f64, n_tilings: usize) -> bool {
            bin_state(&[state], n_tilings)[0] == (state * n_tilings as f64).floor() as isize
        }
    }

    #[test]
    fn test_bin_state_2d() {
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
