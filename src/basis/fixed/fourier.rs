use crate::{
    basis::Projector,
    core::Features,
    geometry::{
        continuous::Interval,
        product::LinearSpace,
        BoundedSpace,
        Card,
        Space,
        Vector,
    },
    utils::cartesian_product,
};
use std::f64::consts::PI;

// TODO: Add support for i-th term alphas scale factors.
/// Fourier basis projector.
///
/// # References
/// - [Konidaris, George, Sarah Osentoski, and Philip S. Thomas. "Value
/// function approximation in reinforcement learning using the Fourier basis."
/// AAAI. Vol. 6. 2011.](http://lis.csail.mit.edu/pubs/konidaris-aaai11a.pdf)
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Clone, Debug)]
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

    pub fn from_space(order: u8, input_space: LinearSpace<Interval>) -> Self {
        Fourier::new(
            order,
            input_space
                .iter()
                .map(|d| (d.inf().unwrap(), d.sup().unwrap()))
                .collect(),
        )
    }

    fn compute_coefficients(order: u8, dim: usize) -> Vec<Vec<f64>> {
        let mut coefficients = cartesian_product(&vec![
            (0..(order + 1)).map(|v| v as f64).collect::<Vec<f64>>(); dim
        ])
        .split_off(1);

        coefficients.sort_by(|a, b| b.partial_cmp(a).unwrap());
        coefficients.dedup();

        coefficients
    }
}

impl Space for Fourier {
    type Value = Features;

    fn dim(&self) -> usize { self.coefficients.len() }

    fn card(&self) -> Card { Card::Infinite }
}

impl Projector<[f64]> for Fourier {
    fn project(&self, input: &[f64]) -> Features {
        let scaled_state = input
            .iter()
            .enumerate()
            .map(|(i, v)| (v - self.limits[i].0) / (self.limits[i].1 - self.limits[i].0))
            .collect::<Vec<f64>>();

        Features::Dense(
            self.coefficients
                .iter()
                .map(|cfs| {
                    let cx = scaled_state
                        .iter()
                        .zip(cfs)
                        .fold(0.0, |acc, (v, c)| acc + *c * v);

                    ((PI * cx).cos() + 1.0) / 2.0
                })
                .collect(),
        )
    }
}

impl_array_proxies!(Fourier; f64);

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_dim() {
        assert_eq!(Fourier::new(1, vec![(0.0, 1.0)]).dim(), 1);
        assert_eq!(Fourier::new(2, vec![(0.0, 1.0)]).dim(), 2);
        assert_eq!(Fourier::new(3, vec![(0.0, 1.0)]).dim(), 3);
        assert_eq!(Fourier::new(4, vec![(0.0, 1.0)]).dim(), 4);
        assert_eq!(Fourier::new(5, vec![(0.0, 1.0)]).dim(), 5);
    }

    #[test]
    fn test_bounds() {
        let f = Fourier::new(1, vec![(0.0, 1.0)]);

        assert_eq!(
            f.project_expanded(&vec![-2.0]),
            f.project_expanded(&vec![0.0]),
        );
        assert_eq!(
            f.project_expanded(&vec![2.0]),
            f.project_expanded(&vec![0.0]),
        );
        assert_eq!(
            f.project_expanded(&vec![-1.0]),
            f.project_expanded(&vec![1.0]),
        );
        assert_eq!(
            f.project_expanded(&vec![3.0]),
            f.project_expanded(&vec![1.0]),
        );
    }

    #[test]
    fn test_order1_1d() {
        let f = Fourier::new(1, vec![(0.0, 1.0)]);

        assert_eq!(f.dim(), 1);
        assert_eq!(f.card(), Card::Infinite);

        assert!(f
            .project_expanded(&vec![0.0])
            .all_close(&arr1(&vec![1.0]), 1e-6));
        assert!(f
            .project_expanded(&vec![1.0 / 3.0])
            .all_close(&arr1(&vec![0.75]), 1e-6));
        assert!(f
            .project_expanded(&vec![0.5])
            .all_close(&arr1(&vec![0.5]), 1e-6));
        assert!(f
            .project_expanded(&vec![2.0 / 3.0])
            .all_close(&arr1(&vec![0.25]), 1e-6));
        assert!(f
            .project_expanded(&vec![1.0])
            .all_close(&arr1(&vec![0.0]), 1e-6));
    }

    #[test]
    fn test_order2_1d() {
        let f = Fourier::new(2, vec![(0.0, 1.0)]);

        assert_eq!(f.dim(), 2);
        assert_eq!(f.card(), Card::Infinite);

        assert!(f
            .project_expanded(&vec![0.0])
            .all_close(&arr1(&vec![1.0, 1.0]), 1e-6));
        assert!(f
            .project_expanded(&vec![1.0 / 3.0])
            .all_close(&arr1(&vec![0.25, 0.75]), 1e-6));
        assert!(f
            .project_expanded(&vec![0.5])
            .all_close(&arr1(&vec![0.0, 0.5]), 1e-6));
        assert!(f
            .project_expanded(&vec![2.0 / 3.0])
            .all_close(&arr1(&vec![0.25, 0.25]), 1e-6));
        assert!(f
            .project_expanded(&vec![1.0])
            .all_close(&arr1(&vec![1.0, 0.0]), 1e-6));
    }

    #[test]
    fn test_order1_2d() {
        let f = Fourier::new(1, vec![(0.0, 1.0), (5.0, 6.0)]);

        assert_eq!(f.dim(), 3);
        assert_eq!(f.card(), Card::Infinite);

        assert!(f
            .project_expanded(&vec![0.0, 5.0])
            .all_close(&arr1(&vec![1.0; 3]), 1e-6));
        assert!(f
            .project_expanded(&vec![0.5, 5.0])
            .all_close(&arr1(&vec![0.5, 0.5, 1.0]), 1e-6,));
        assert!(f
            .project_expanded(&vec![0.0, 5.5])
            .all_close(&arr1(&vec![0.5, 1.0, 0.5]), 1e-6,));
        assert!(f
            .project_expanded(&vec![0.5, 5.5])
            .all_close(&arr1(&vec![0.0, 0.5, 0.5]), 1e-6));
        assert!(f
            .project_expanded(&vec![1.0, 5.5])
            .all_close(&arr1(&vec![0.5, 0.0, 0.5]), 1e-6,));
        assert!(f
            .project_expanded(&vec![0.5, 6.0])
            .all_close(&arr1(&vec![0.5, 0.5, 0.0]), 1e-6,));
        assert!(f
            .project_expanded(&vec![1.0, 6.0])
            .all_close(&arr1(&vec![1.0, 0.0, 0.0]), 1e-6));
    }

    #[test]
    fn test_order2_2d() {
        let f = Fourier::new(2, vec![(0.0, 1.0), (5.0, 6.0)]);

        assert_eq!(f.dim(), 8);
        assert_eq!(f.card(), Card::Infinite);

        assert!(f
            .project_expanded(&vec![0.0, 5.0])
            .all_close(&arr1(&vec![1.0; 8]), 1e-6));
        assert!(f.project_expanded(&vec![0.5, 5.0],).all_close(
            &arr1(&vec![0.0, 0.0, 0.0, 0.5, 0.5, 0.5, 1.0, 1.0]),
            1e-6,
        ));
        assert!(f.project_expanded(&vec![0.0, 5.5],).all_close(
            &arr1(&vec![0.0, 0.5, 1.0, 0.0, 0.5, 1.0, 0.0, 0.5]),
            1e-6,
        ));
        assert!(f.project_expanded(&vec![0.5, 5.5],).all_close(
            &arr1(&vec![1.0, 0.5, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5]),
            1e-6,
        ));
        assert!(f.project_expanded(&vec![0.5, 6.0],).all_close(
            &arr1(&vec![0.0, 1.0, 0.0, 0.5, 0.5, 0.5, 1.0, 0.0]),
            1e-6,
        ));
        assert!(f.project_expanded(&vec![1.0, 6.0],).all_close(
            &arr1(&vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0]),
            1e-6,
        ));
    }
}
