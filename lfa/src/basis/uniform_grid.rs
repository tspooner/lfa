use crate::{IndexT, ActivationT, Features, Result, check_index, basis::Projector};
use spaces::{Space, Surjection, Equipartition, ProductSpace};

/// Fixed uniform basis projector.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
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

    fn hash(&self, input: &[f64]) -> usize {
        let mut in_it = input.iter().rev();
        let mut d_it = self.feature_space.iter().rev();

        let acc = d_it.next().unwrap().map_onto(*in_it.next().unwrap());

        d_it.zip(in_it).fold(acc, |acc, (d, v)| d.map_onto(*v) + d.n_partitions() * acc)
    }
}

impl Projector for UniformGrid {
    fn n_features(&self) -> usize { self.n_features }

    fn project_ith(&self, input: &[f64], index: IndexT) -> Result<Option<ActivationT>> {
        check_index(index, self.n_features, || Ok(if self.hash(input) == index {
            Some(1.0)
        } else {
            None
        }))
    }

    fn project(&self, input: &[f64]) -> Result<Features> {
        let active_bin = self.hash(input);

        Ok(Features::Sparse(self.n_features, ::std::iter::once((active_bin, 1.0)).collect()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_1d() {
        let ds = ProductSpace::new(vec![Equipartition::new(0.0, 10.0, 10)]);
        let t = UniformGrid::new(ds);

        assert_eq!(t.n_features(), 10);

        for bin in 0..10 {
            let fv = t.project(&vec![bin as u32 as f64]).unwrap();

            assert_eq!(fv.n_features(), 10);
            assert_eq!(fv.n_active(), 1);

            unsafe {
                assert!(fv.uget(bin).filter(|&&a| a > 0.0).is_some());
            }

            let mut dense = vec![0.0; 10];
            dense[bin] = 1.0;

            assert_eq!(dense, fv.expanded().into_raw_vec());
        }
    }

    #[test]
    fn test_2d() {
        let ds = ProductSpace::new(vec![Equipartition::new(0.0, 10.0, 10); 2]);
        let t = UniformGrid::new(ds);

        assert_eq!(t.n_features(), 100);

        for bin_i in 0..10 {
            for bin_j in 0..10 {
                let fv = t.project(&vec![bin_i as u32 as f64, bin_j as u32 as f64]).unwrap();
                let active_bin = bin_j * 10 + bin_i;

                assert_eq!(fv.n_features(), 100);
                assert_eq!(fv.n_active(), 1);

                unsafe {
                    assert!(fv.uget(active_bin).filter(|&&a| a > 0.0).is_some());
                }

                let mut dense = vec![0.0; 100];
                dense[active_bin] = 1.0;

                assert_eq!(dense, fv.expanded().into_raw_vec());
            }
        }
    }

    #[test]
    fn test_3d() {
        let ds = ProductSpace::new(vec![Equipartition::new(0.0, 10.0, 10); 3]);
        let t = UniformGrid::new(ds);

        assert_eq!(t.n_features(), 1000);

        for bin_i in 0..10 {
            for bin_j in 0..10 {
                for bin_k in 0..10 {
                    let fv = t.project(&vec![
                        bin_i as u32 as f64,
                        bin_j as u32 as f64,
                        bin_k as u32 as f64
                    ]).unwrap();
                    let active_bin = bin_k * 100 + bin_j * 10 + bin_i;

                    assert_eq!(fv.n_features(), 1000);
                    assert_eq!(fv.n_active(), 1);

                    unsafe {
                        assert!(fv.uget(active_bin).filter(|&&a| a > 0.0).is_some());
                    }

                    let mut dense = vec![0.0; 1000];
                    dense[active_bin] = 1.0;

                    assert_eq!(dense, fv.expanded().into_raw_vec());
                }
            }
        }
    }
}
