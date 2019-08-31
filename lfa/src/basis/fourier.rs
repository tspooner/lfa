use crate::{
    IndexT, ActivationT, Features, Result, Error,
    basis::Projector,
};
use spaces::{ProductSpace, real::Interval};
use std::f64::consts::PI;
use super::{compute_coefficients, get_bounds};

/// Fourier basis projector.
///
/// # References
/// - [Konidaris, George, Sarah Osentoski, and Philip S. Thomas. "Value
/// function approximation in reinforcement learning using the Fourier basis."
/// AAAI. Vol. 6. 2011.](http://lis.csail.mit.edu/pubs/konidaris-aaai11a.pdf)
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
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
        Fourier::new(
            order,
            input_space.iter().map(get_bounds).collect(),
        )
    }

    fn rescale_input(&self, input: &[f64]) -> Vec<f64> {
        input
            .iter()
            .enumerate()
            .map(|(i, v)| (v - self.limits[i].0) / (self.limits[i].1 - self.limits[i].0))
            .collect::<Vec<f64>>()
    }

    fn compute_feature(&self, ss: &[f64], cfs: &[f64]) -> f64 {
        let cx = ss.iter().zip(cfs).fold(0.0, |acc, (v, c)| acc + *c * v);

        (PI * cx).cos()
    }
}

impl Projector for Fourier {
    fn n_features(&self) -> usize { self.coefficients.len() }

    fn project_ith(&self, input: &[f64], index: IndexT) -> Result<Option<ActivationT>> {
        self.coefficients.get(index).map(|cfs| {
            let scaled_state = self.rescale_input(input);

            Some(self.compute_feature(&scaled_state, cfs))
        }).ok_or_else(|| Error::index_error(index, self.n_features()))
    }

    fn project(&self, input: &[f64]) -> Result<Features> {
        let scaled_state = self.rescale_input(input);

        Ok(self.coefficients.iter().map(|cfs| self.compute_feature(&scaled_state, cfs)).collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_n_features() {
        fn get_n(order: u8) -> usize {
            Fourier::new(order, vec![(0.0, 1.0)]).n_features()
        }

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

        assert_eq!(f.n_features(), 1);
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

        assert_eq!(f.n_features(), 2);
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

        assert_eq!(f.n_features(), 3);
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

        assert_eq!(f.n_features(), 8);
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
