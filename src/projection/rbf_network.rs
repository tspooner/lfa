use super::{Projection, Projector, Vector};

/// Radial basis function network projector.
#[derive(Clone, Serialize, Deserialize)]
pub struct RBFNetwork {
    mu: Vec<Vec<f64>>,
    beta: Vec<f64>,
}

impl RBFNetwork {
    pub fn new(mu: Vec<Vec<f64>>, sigma: Vec<f64>) -> Self {
        let n_dims = mu[0].len();
        if !mu.iter().skip(1).all(|ref x| x.len() == n_dims) {
            panic!("Rows of mu must all have the same length.");
        }

        if sigma.len() != n_dims {
            panic!(
                "Dimensionality of mu ({}) and sigma ({}) must agree.",
                n_dims,
                sigma.len()
            );
        }

        RBFNetwork {
            mu: mu,
            beta: sigma.iter().map(|s| 1.0 / (2.0 * s * s)).collect(),
        }
    }

    pub fn kernel(&self, input: &[f64]) -> Vector<f64> {
        self.mu
            .iter()
            .map(|ref mu| {
                mu.iter()
                    .zip(input.iter())
                    .map(|(c, v)| c - v)
                    .zip(self.beta.iter())
                    .fold(0.0, |acc, (d, b)| acc + b * d * d)
            })
            .map(|exponent| (-exponent.abs()).exp())
            .collect()
    }
}

impl Projector<[f64]> for RBFNetwork {
    fn project(&self, input: &[f64]) -> Projection { Projection::Dense(self.kernel(input)) }

    fn dim(&self) -> usize { self.mu[0].len() }

    fn size(&self) -> usize { self.mu.len() }

    fn activity(&self) -> usize { self.size() }

    fn equivalent(&self, other: &Self) -> bool {
        self.mu == other.mu && self.beta == other.beta && self.size() == other.size()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use utils::vec_compare;

    #[test]
    fn test_size() {
        assert_eq!(RBFNetwork::new(vec![vec![0.0]], vec![0.25]).size(), 1);
        assert_eq!(
            RBFNetwork::new(vec![vec![0.0], vec![0.5], vec![1.0]], vec![0.25]).size(),
            3
        );
        assert_eq!(RBFNetwork::new(vec![vec![0.0]; 10], vec![0.25]).size(), 10);
        assert_eq!(
            RBFNetwork::new(vec![vec![0.0]; 100], vec![0.25]).size(),
            100
        );
    }

    #[test]
    fn test_dimensionality() {
        assert_eq!(
            RBFNetwork::new(vec![vec![0.0], vec![0.5], vec![1.0]], vec![0.25]).dim(),
            1
        );
        assert_eq!(
            RBFNetwork::new(vec![vec![0.0, 0.5, 1.0]; 10], vec![0.1, 0.2, 0.3]).dim(),
            3
        );
    }

    #[test]
    fn test_kernel_relevance() {
        let rbf = RBFNetwork::new(vec![vec![0.0]], vec![0.25]);
        let mut p = rbf.kernel(&[0.0]);

        for i in 1..10 {
            let p_new = rbf.kernel(&[i as f64 / 10.0]);
            assert!(p_new[0] < p[0]);

            p = p_new
        }
    }

    #[test]
    fn test_kernel_isotropy() {
        let rbf = RBFNetwork::new(vec![vec![0.0]], vec![0.25]);
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
        let rbf = RBFNetwork::new(vec![vec![0.0], vec![0.5], vec![1.0]], vec![0.25]);
        let p = rbf.project_expanded(&vec![0.25]);

        assert!(vec_compare(
            &p.as_slice().unwrap(),
            &[0.49546264, 0.49546264, 0.00907471],
            1e-6
        ));
        assert_eq!(p.iter().fold(0.0, |acc, x| acc + *x), 1.0);
    }

    #[test]
    fn test_projection_2d() {
        let rbf = RBFNetwork::new(
            vec![vec![0.0, -10.0], vec![0.5, -8.0], vec![1.0, -6.0]],
            vec![0.25, 2.0],
        );
        let p = rbf.project_expanded(&[0.67, -7.0]);

        assert!(vec_compare(
            &p.as_slice().unwrap(),
            &[0.00829727, 0.64932079, 0.34238193],
            1e-6
        ));
        assert_eq!(p.iter().fold(0.0, |acc, x| acc + *x), 1.0);
    }
}
