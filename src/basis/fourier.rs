use super::*;
use crate::{ActivationT, Error, Features, IndexT, Result};
use spaces::{real::Interval, Card, Dim, ProductSpace, Space};
use std::f64::consts::PI;

/// Fourier basis projector.
///
/// # References
/// - [Konidaris, George, Sarah Osentoski, and Philip S. Thomas. "Value
/// function approximation in reinforcement learning using the Fourier basis."
/// AAAI. Vol. 6. 2011.](http://lis.csail.mit.edu/pubs/konidaris-aaai11a.pdf)
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Fourier {
    pub order: u8,
    pub limits: Vec<(f64, f64)>,
    pub coefficients: Vec<Vec<f64>>,
}

impl Fourier {
    pub fn new(order: u8, limits: Vec<(f64, f64)>) -> Self {
        let coefficients = compute_coefficients(order, limits.len())
            .map(|cfs| cfs.into_iter().map(|c| c as f64).collect())
            .collect();

        Fourier {
            order,
            limits,
            coefficients,
        }
    }

    pub fn from_space(order: u8, input_space: ProductSpace<Interval>) -> Self {
        Fourier::new(order, input_space.iter().map(get_bounds).collect())
    }

    #[inline]
    fn compute_feature<'a, I>(iter: I) -> f64
    where I: IntoIterator<Item = (f64, &'a f64)> {
        (PI * iter.into_iter().fold(0.0, |acc, (v, c)| acc + c * v)).cos()
    }
}

impl Space for Fourier {
    type Value = Features;

    fn dim(&self) -> Dim { Dim::Finite(self.coefficients.len()) }

    fn card(&self) -> Card { Card::Infinite }
}

impl<I: std::borrow::Borrow<f64>, T: IntoIterator<Item = I>> Basis<T> for Fourier {
    fn project(&self, input: T) -> Result<Features> {
        let ss: Vec<f64> = rescale!(input into self.limits).collect();

        Ok(self
            .coefficients
            .iter()
            .map(|cfs| Self::compute_feature(ss.iter().copied().zip(cfs.iter())))
            .collect())
    }
}

impl<I: std::borrow::Borrow<f64>, T: IntoIterator<Item = I>> EnumerableBasis<T> for Fourier {
    fn ith(&self, input: T, index: IndexT) -> Result<ActivationT> {
        self.coefficients
            .get(index)
            .map(|cfs| {
                let ss = rescale!(input into self.limits);

                Self::compute_feature(ss.zip(cfs.iter()))
            })
            .ok_or_else(|| Error::index_error(index, self.dim().into()))
    }
}

impl Combinators for Fourier {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_n_features() {
        fn get_n(order: u8) -> usize { Fourier::new(order, vec![(0.0, 1.0)]).dim().into() }

        assert_eq!(get_n(1), 1);
        assert_eq!(get_n(2), 2);
        assert_eq!(get_n(3), 3);
        assert_eq!(get_n(4), 4);
        assert_eq!(get_n(5), 5);
    }

    #[test]
    fn test_bounds() {
        let f = Fourier::new(1, vec![(0.0, 1.0)]);

        assert_features!(f +- 1e-6 [
            vec![-2.0] => vec![1.0],
            vec![2.0] => vec![1.0],
            vec![-1.0] => vec![-1.0],
            vec![3.0] => vec![-1.0]
        ]);
    }

    #[test]
    fn test_order1_1d() {
        let f = Fourier::new(1, vec![(0.0, 1.0)]);
        let nf: usize = f.dim().into();

        assert_eq!(nf, 1);
        assert_features!(f +- 1e-6 [
            vec![0.0]       => vec![1.0],
            vec![1.0 / 3.0] => vec![0.5],
            vec![0.5]       => vec![0.0],
            vec![2.0 / 3.0] => vec![-0.5],
            vec![1.0]       => vec![-1.0]
        ]);
    }

    #[test]
    fn test_order2_1d() {
        let f = Fourier::new(2, vec![(0.0, 1.0)]);
        let nf: usize = f.dim().into();

        assert_eq!(nf, 2);
        assert_features!(f +- 1e-6 [
            vec![0.0]       => vec![1.0, 1.0],
            vec![1.0 / 3.0] => vec![0.5, -0.5],
            vec![0.5]       => vec![0.0, -1.0],
            vec![2.0 / 3.0] => vec![-0.5, -0.5],
            vec![1.0]       => vec![-1.0, 1.0]
        ]);
    }

    #[test]
    fn test_order1_2d() {
        let f = Fourier::new(1, vec![(0.0, 1.0), (5.0, 6.0)]);
        let nf: usize = f.dim().into();

        assert_eq!(nf, 3);
        assert_features!(f +- 1e-6 [
            vec![0.0, 5.0] => vec![1.0; 3],
            vec![0.5, 5.0] => vec![1.0, 0.0, 0.0],
            vec![0.0, 5.5] => vec![0.0, 1.0, 0.0],
            vec![0.5, 5.5] => vec![0.0, 0.0, -1.0],
            vec![1.0, 5.5] => vec![0.0, -1.0, 0.0],
            vec![0.5, 6.0] => vec![-1.0, 0.0, 0.0],
            vec![1.0, 6.0] => vec![-1.0, -1.0, 1.0]
        ]);
    }

    #[test]
    fn test_order2_2d() {
        let f = Fourier::new(2, vec![(0.0, 1.0), (5.0, 6.0)]);
        let nf: usize = f.dim().into();

        assert_eq!(nf, 8);
        assert_features!(f +- 1e-6 [
            vec![0.0, 5.0] => vec![1.0; 8],
            vec![0.5, 5.0] => vec![1.0, 1.0, 0.0, 0.0, 0.0, -1.0, -1.0, -1.0],
            vec![0.0, 5.5] => vec![0.0, -1.0, 1.0, 0.0, -1.0, 1.0, 0.0, -1.0],
            vec![0.5, 5.5] => vec![0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0],
            // vec![1.0, 5.5] => vec![],
            vec![0.5, 6.0] => vec![-1.0, 1.0, 0.0, 0.0, 0.0, -1.0, 1.0, -1.0],
            vec![1.0, 6.0] => vec![-1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]
        ]);
    }
}
