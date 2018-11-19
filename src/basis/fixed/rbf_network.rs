use basis::{Projection, Projector};
use geometry::{Card, Matrix, product::RegularSpace, Space, Vector, discrete::Partition};
use ndarray::Axis;
use rand::Rng;
use utils::cartesian_product;

/// Radial basis function network projector.
#[derive(Clone, Serialize, Deserialize)]
pub struct RBFNetwork {
    pub mu: Matrix<f64>,
    pub beta: Vector<f64>,
}

impl RBFNetwork {
    pub fn new(mu: Matrix<f64>, sigma: Vector<f64>) -> Self {
        if sigma.len() != mu.cols() {
            panic!(
                "Dimensionality of mu ({}) and sigma ({}) must agree.",
                mu.cols(),
                sigma.len()
            );
        }

        RBFNetwork {
            mu: mu,
            beta: sigma.iter().map(|s| 1.0 / (2.0 * s * s)).collect(),
        }
    }

    pub fn from_space(input_space: RegularSpace<Partition>) -> Self {
        let n_features = match input_space.card() {
            Card::Finite(s) => s,
            _ => panic!("`RBFNetwork` projection only supports partitioned input spaces."),
        };

        let centres = input_space.centres();
        let flat_combs = cartesian_product(&centres)
            .iter()
            .cloned()
            .flat_map(|e| e)
            .collect();

        let mu = Matrix::from_shape_vec((n_features, input_space.dim()), flat_combs).unwrap();
        let sigma = input_space.iter().map(|d| d.partition_width()).collect();

        RBFNetwork::new(mu, sigma)
    }

    pub fn kernel(&self, input: &[f64]) -> Vector<f64> {
        self.mu
            .axis_iter(Axis(0))
            .map(|col| {
                col.iter()
                    .zip(input.iter())
                    .map(|(c, v)| c - v)
                    .zip(self.beta.iter())
                    .fold(0.0, |acc, (d, b)| acc + b * d * d)
            })
            .map(|exponent| (-exponent.abs()).exp())
            .collect()
    }
}

impl Space for RBFNetwork {
    type Value = Projection;

    fn sample<R: Rng + ?Sized>(&self, _: &mut R) -> Projection {
        unimplemented!()
    }

    fn dim(&self) -> usize {
        self.mu.rows()
    }

    fn card(&self) -> Card {
        Card::Infinite
    }
}

impl Projector<[f64]> for RBFNetwork {
    fn project(&self, input: &[f64]) -> Projection {
        Projection::Dense(self.kernel(input))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr1, arr2};

    #[test]
    fn test_dimensionality() {
        assert_eq!(RBFNetwork::new(arr2(&[[0.0]]), arr1(&[0.25])).dim(), 1);
        assert_eq!(RBFNetwork::new(arr2(&[[0.0], [0.5], [1.0]]), arr1(&[0.25])).dim(), 3);
        assert_eq!(RBFNetwork::new(arr2(&vec![[0.0]; 10]), arr1(&[0.25])).dim(), 10);
        assert_eq!(RBFNetwork::new(arr2(&vec![[0.0]; 100]), arr1(&[0.25])).dim(), 100);
    }

    #[test]
    fn test_cardinality() {
        assert_eq!(
            RBFNetwork::new(arr2(&[[0.0]]), arr1(&[0.25])).card(),
            Card::Infinite
        );
        assert_eq!(
            RBFNetwork::new(arr2(&[[0.0], [0.5], [1.0]]), arr1(&[0.25])).card(),
            Card::Infinite
        );
        assert_eq!(
            RBFNetwork::new(arr2(&vec![[0.0]; 10]), arr1(&[0.25])).card(),
            Card::Infinite
        );
        assert_eq!(
            RBFNetwork::new(arr2(&vec![[0.0]; 100]), arr1(&[0.25])).card(),
            Card::Infinite
        );
    }

    #[test]
    fn test_kernel_relevance() {
        let rbf = RBFNetwork::new(arr2(&[[0.0]]), arr1(&[0.25]));
        let mut p = rbf.kernel(&[0.0]);

        for i in 1..10 {
            let p_new = rbf.kernel(&[i as f64 / 10.0]);
            assert!(p_new[0] < p[0]);

            p = p_new
        }
    }

    #[test]
    fn test_kernel_isotropy() {
        let rbf = RBFNetwork::new(arr2(&[[0.0]]), arr1(&[0.25]));
        let p = rbf.kernel(&[0.0]);

        for i in 1..10 {
            let p_left = rbf.kernel(&[-i as f64 / 10.0]);
            let p_right = rbf.kernel(&[i as f64 / 10.0]);

            assert!(p_left[0] < p[0]);
            assert!(p_right[0] < p[0]);
            assert_eq!(p_left, p_right);
        }
    }

    #[test]
    fn test_projection_1d() {
        let rbf = RBFNetwork::new(arr2(&[[0.0], [0.5], [1.0]]), arr1(&[0.25]));
        let p = rbf.project_expanded(&[0.25]);

        assert!(p.all_close(&arr1(&[0.6065307, 0.6065307, 0.0111090]), 1e-6));
    }

    #[test]
    fn test_projection_2d() {
        let rbf = RBFNetwork::new(
            arr2(&[[0.0, -10.0], [0.5, -8.0], [1.0, -6.0]]),
            arr1(&[0.25, 2.0]),
        );
        let p = rbf.project_expanded(&[0.67, -7.0]);

        assert!(p.all_close(&arr1(&[0.0089491, 0.7003325, 0.369280]), 1e-6));
    }
}
