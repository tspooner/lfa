use super::*;
use crate::{ActivationT, Features, IndexT, Result, SparseActivations};
use spaces::{Card, Dim, Equipartition, ProductSpace, Space, Surjection};

/// Fixed uniform basis.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct UniformGrid {
    n_features: usize,
    feature_space: ProductSpace<Equipartition>,
}

impl UniformGrid {
    pub fn new(feature_space: ProductSpace<Equipartition>) -> Self {
        let n_features = feature_space.card().into();

        UniformGrid {
            n_features,
            feature_space,
        }
    }

    fn hash<'a, T>(&self, input: T) -> usize
    where T: IntoIterator<Item = &'a f64> {
        let mut in_it = input.into_iter().copied();
        let mut d_it = self.feature_space.iter();

        let acc = d_it.next().unwrap().map_onto(in_it.next().unwrap());

        d_it.zip(in_it)
            .fold(acc, |acc, (d, v)| d.map_onto(v) + d.n_partitions() * acc)
    }
}

impl Space for UniformGrid {
    type Value = Features;

    fn dim(&self) -> Dim { Dim::Finite(self.n_features) }

    fn card(&self) -> Card { Card::Infinite }
}

impl<'a, T> Basis<T> for UniformGrid
where T: IntoIterator<Item = &'a f64>
{
    fn project(&self, input: T) -> Result<Features> {
        let active_bin = self.hash(input);

        Ok(Features::Sparse(SparseActivations {
            dim: self.n_features,
            activations: ::std::iter::once((active_bin, 1.0)).collect(),
        }))
    }
}

impl<'a, T> EnumerableBasis<T> for UniformGrid
where T: IntoIterator<Item = &'a f64>
{
    fn ith(&self, input: T, index: IndexT) -> Result<ActivationT> {
        check_index!(index < self.n_features => {
            Ok(if self.hash(input) == index { 1.0 } else { 0.0 })
        })
    }
}

impl Combinators for UniformGrid {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d() {
        let ds = ProductSpace::new(vec![Equipartition::new(0.0, 10.0, 10)]);
        let t = UniformGrid::new(ds);
        let nf: usize = t.dim().into();

        assert_eq!(nf, 10);

        for bin in 0..10 {
            let fv = t.project(&[bin as u32 as f64]).unwrap();

            assert_eq!(fv.n_features(), 10);
            assert_eq!(fv.n_active(), 1);

            unsafe {
                assert!(fv.uget(bin).filter(|&&a| a > 0.0).is_some());
            }

            let mut dense = vec![0.0; 10];
            dense[bin] = 1.0;

            assert_eq!(dense, fv.into_dense().into_raw_vec());
        }
    }

    #[test]
    fn test_2d() {
        let ds = ProductSpace::new(vec![Equipartition::new(0.0, 10.0, 10); 2]);
        let t = UniformGrid::new(ds);
        let nf: usize = t.dim().into();

        assert_eq!(nf, 100);

        for bin_i in 0..10 {
            for bin_j in 0..10 {
                let fv = t
                    .project(&[bin_i as u32 as f64, bin_j as u32 as f64])
                    .unwrap();
                let active_bin = bin_i * 10 + bin_j;

                assert_eq!(fv.n_features(), 100);
                assert_eq!(fv.n_active(), 1);

                unsafe {
                    assert!(fv.uget(active_bin).filter(|&&a| a > 0.0).is_some());
                }

                let mut dense = vec![0.0; 100];
                dense[active_bin] = 1.0;

                assert_eq!(dense, fv.into_dense().into_raw_vec());
            }
        }
    }

    #[test]
    fn test_3d() {
        let ds = ProductSpace::new(vec![Equipartition::new(0.0, 10.0, 10); 3]);
        let t = UniformGrid::new(ds);
        let nf: usize = t.dim().into();

        assert_eq!(nf, 1000);

        for bin_i in 0..10 {
            for bin_j in 0..10 {
                for bin_k in 0..10 {
                    let fv = t
                        .project(&[
                            bin_i as u32 as f64,
                            bin_j as u32 as f64,
                            bin_k as u32 as f64,
                        ])
                        .unwrap();
                    let active_bin = bin_i * 100 + bin_j * 10 + bin_k;

                    assert_eq!(fv.n_features(), 1000);
                    assert_eq!(fv.n_active(), 1);

                    unsafe {
                        assert!(fv.uget(active_bin).filter(|&&a| a > 0.0).is_some());
                    }

                    let mut dense = vec![0.0; 1000];
                    dense[active_bin] = 1.0;

                    assert_eq!(dense, fv.into_dense().into_raw_vec());
                }
            }
        }
    }
}
