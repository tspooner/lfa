use geometry::{BoundedSpace, Card, RegularSpace, Space, dimensions::Continuous};
use projectors::{Projection, Projector};
use rand::{Rng, distributions::{Distribution, Range}};
use std::{
    f64::consts::PI,
    iter,
};
use utils::cartesian_product;

// TODO: Add support for i-th term alphas scale factors.
/// Fourier basis projector.
///
/// # References
/// - [Konidaris, George, Sarah Osentoski, and Philip S. Thomas. "Value
/// function approximation in reinforcement learning using the Fourier basis."
/// AAAI. Vol. 6. 2011.](http://lis.csail.mit.edu/pubs/konidaris-aaai11a.pdf)
#[derive(Clone, Serialize, Deserialize)]
pub struct Fourier {
    pub order: u8,
    pub limits: Vec<(f64, f64)>,
    pub coefficients: Vec<Vec<f64>>,
}

impl Fourier {
    pub fn new(order: u8, limits: Vec<(f64, f64)>) -> Self {
        let coefficients = Fourier::compute_coefficients(order, limits.len());

        Fourier {
            order,
            limits,
            coefficients,
        }
    }

    pub fn from_space(order: u8, input_space: RegularSpace<Continuous>) -> Self {
        Fourier::new(
            order,
            input_space.iter().map(|d| (*d.lb(), *d.ub())).collect(),
        )
    }

    fn compute_coefficients(order: u8, dim: usize) -> Vec<Vec<f64>> {
        let mut coefficients = cartesian_product(&vec![
            (0..(order + 1)).map(|v| v as f64).collect::<Vec<f64>>(); dim
        ]).split_off(1);

        coefficients.sort_by(|a, b| b.partial_cmp(a).unwrap());
        coefficients.dedup();

        coefficients
    }
}

impl Space for Fourier {
    type Value = Projection;

    fn sample<R: Rng + ?Sized>(&self, rng: &mut R) -> Projection {
        let random_input: Vec<f64> = self.limits
            .iter()
            .map(|&(ll, ul)| Range::new(ll, ul).sample(rng))
            .collect();

        self.project(&random_input)
    }

    fn dim(&self) -> usize {
        self.coefficients.len() + 1
    }

    fn card(&self) -> Card {
        Card::Infinite
    }
}

impl Projector<[f64]> for Fourier {
    fn project(&self, input: &[f64]) -> Projection {
        let scaled_state = input
            .iter()
            .enumerate()
            .map(|(i, v)| (v - self.limits[i].0) / (self.limits[i].1 - self.limits[i].0))
            .collect::<Vec<f64>>();

        Projection::Dense(self.coefficients
            .iter()
            .map(|cfs| {
                let cx = scaled_state
                    .iter()
                    .zip(cfs)
                    .fold(0.0, |acc, (v, c)| acc + *c * v);

                (PI * cx).cos()
            })
            .chain(iter::once(1.0))
            .collect()
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_dim() {
        assert_eq!(Fourier::new(1, vec![(0.0, 1.0)]).dim(), 2);
        assert_eq!(Fourier::new(2, vec![(0.0, 1.0)]).dim(), 3);
        assert_eq!(Fourier::new(3, vec![(0.0, 1.0)]).dim(), 4);
        assert_eq!(Fourier::new(4, vec![(0.0, 1.0)]).dim(), 5);
        assert_eq!(Fourier::new(5, vec![(0.0, 1.0)]).dim(), 6);
    }

    #[test]
    fn test_symmetry() {
        let f = Fourier::new(1, vec![(0.0, 1.0)]);

        assert_eq!(
            f.project_expanded(&vec![-1.0]),
            f.project_expanded(&vec![1.0])
        );
        assert_eq!(
            f.project_expanded(&vec![-0.5]),
            f.project_expanded(&vec![0.5])
        );
    }

    #[test]
    fn test_order1_1d() {
        let f = Fourier::new(1, vec![(0.0, 1.0)]);

        assert_eq!(f.dim(), 2);
        assert_eq!(f.card(), Card::Infinite);

        assert!(
            f.project_expanded(&vec![-1.0])
                .all_close(&arr1(&vec![-1.0, 1.0]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![-0.5])
                .all_close(&arr1(&vec![0.0, 1.0]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![0.0])
                .all_close(&arr1(&vec![1.0, 1.0]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![0.5])
                .all_close(&arr1(&vec![0.0, 1.0]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![1.0])
                .all_close(&arr1(&vec![-1.0, 1.0]), 1e-6)
        );

        assert!(
            f.project_expanded(&vec![-2.0 / 3.0])
                .all_close(&arr1(&vec![-0.5, 1.0]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![-1.0 / 3.0])
                .all_close(&arr1(&vec![0.5, 1.0]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![1.0 / 3.0])
                .all_close(&arr1(&vec![0.5, 1.0]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![2.0 / 3.0])
                .all_close(&arr1(&vec![-0.5, 1.0]), 1e-6)
        );
    }

    #[test]
    fn test_order2_1d() {
        let f = Fourier::new(2, vec![(0.0, 1.0)]);

        assert_eq!(f.dim(), 3);
        assert_eq!(f.card(), Card::Infinite);

        assert!(
            f.project_expanded(&vec![-1.0])
                .all_close(&arr1(&vec![1.0, -1.0, 1.0]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![-0.5])
                .all_close(&arr1(&vec![-1.0, 0.0, 1.0]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![0.0])
                .all_close(&arr1(&vec![1.0; 3]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![0.5])
                .all_close(&arr1(&vec![-1.0, 0.0, 1.0]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![1.0])
                .all_close(&arr1(&vec![1.0, -1.0, 1.0]), 1e-6)
        );

        assert!(
            f.project_expanded(&vec![-2.0 / 3.0])
                .all_close(&arr1(&vec![-0.5, -0.5, 1.0]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![-1.0 / 3.0])
                .all_close(&arr1(&vec![-0.5, 0.5, 1.0]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![1.0 / 3.0])
                .all_close(&arr1(&vec![-0.5, 0.5, 1.0]), 1e-6)
        );
        assert!(
            f.project_expanded(&vec![2.0 / 3.0])
                .all_close(&arr1(&vec![-0.5, -0.5, 1.0]), 1e-6)
        );
    }

    #[test]
    fn test_order1_2d() {
        let f = Fourier::new(1, vec![(0.0, 1.0), (5.0, 6.0)]);

        assert_eq!(f.dim(), 4);
        assert_eq!(f.card(), Card::Infinite);

        assert!(f.project_expanded(&vec![0.0, 5.0]).all_close(
            &arr1(&vec![1.0; 4]),
            1e-6
        ));
        assert!(f.project_expanded(&vec![0.5, 5.0]).all_close(
            &arr1(&vec![0.0, 0.0, 1.0, 1.0]),
            1e-6,
        ));
        assert!(f.project_expanded(&vec![0.0, 5.5]).all_close(
            &arr1(&vec![0.0, 1.0, 0.0, 1.0]),
            1e-6,
        ));
        assert!(f.project_expanded(&vec![0.5, 5.5]).all_close(
            &arr1(&vec![-1.0, 0.0, 0.0, 1.0]),
            1e-6
        ));
        assert!(f.project_expanded(&vec![1.0, 5.5]).all_close(
            &arr1(&vec![0.0, -1.0, 0.0, 1.0]),
            1e-6,
        ));
        assert!(f.project_expanded(&vec![0.5, 6.0]).all_close(
            &arr1(&vec![0.0, 0.0, -1.0, 1.0]),
            1e-6,
        ));
        assert!(f.project_expanded(&vec![1.0, 6.0]).all_close(
            &arr1(&vec![1.0, -1.0, -1.0, 1.0]),
            1e-6
        ));
    }

    #[test]
    fn test_order2_2d() {
        let f = Fourier::new(2, vec![(0.0, 1.0), (5.0, 6.0)]);

        assert_eq!(f.dim(), 9);
        assert_eq!(f.card(), Card::Infinite);

        assert!(f.project_expanded(&vec![0.0, 5.0]).all_close(
            &arr1(&vec![1.0; 9]),
            1e-6
        ));
        assert!(f.project_expanded(&vec![0.5, 5.0],).all_close(
            &arr1(&vec![-1.0, -1.0, -1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
            1e-6,
        ));
        assert!(f.project_expanded(&vec![0.0, 5.5],).all_close(
            &arr1(&vec![-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0]),
            1e-6,
        ));
        assert!(f.project_expanded(&vec![0.5, 5.5],).all_close(
            &arr1(&vec![1.0, 0.0, -1.0, 0.0, -1.0, 0.0, -1.0, 0.0, 1.0]),
            1e-6,
        ));
        assert!(f.project_expanded(&vec![0.5, 6.0],).all_close(
            &arr1(&vec![-1.0, 1.0, -1.0, 0.0, 0.0, 0.0, 1.0, -1.0, 1.0]),
            1e-6,
        ));
        assert!(f.project_expanded(&vec![1.0, 6.0],).all_close(
            &arr1(&vec![1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]),
            1e-6,
        ));
    }
}
