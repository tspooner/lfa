use super::{Projection, Projector};
use utils::cartesian_product;

/// Polynomial basis projector.
#[derive(Clone, Serialize, Deserialize)]
pub struct Polynomial {
    order: u8,
    limits: Vec<(f64, f64)>,
    exponents: Vec<Vec<i32>>,
}

impl Polynomial {
    pub fn new(order: u8, limits: Vec<(f64, f64)>) -> Self {
        let exponents = Polynomial::make_exponents(order, limits.len());

        Polynomial {
            order: order,
            limits: limits,
            exponents: exponents,
        }
    }

    fn make_exponents(order: u8, dim: usize) -> Vec<Vec<i32>> {
        let dcs = vec![(0..(order + 1)).map(|v| v as i32).collect::<Vec<i32>>(); dim];
        let mut exponents = cartesian_product(&dcs);

        exponents.sort_by(|a, b| b.partial_cmp(a).unwrap());
        exponents.dedup();

        exponents
    }
}

impl Projector<[f64]> for Polynomial {
    fn project(&self, input: &[f64]) -> Projection {
        let scaled_state = input
            .iter()
            .enumerate()
            .map(|(i, v)| (v - self.limits[i].0) / (self.limits[i].1 - self.limits[i].0))
            .map(|v| 2.0 * v - 1.0)
            .collect::<Vec<f64>>();

        let activations = self.exponents.iter().map(|exps| {
            scaled_state
                .iter()
                .zip(exps)
                .map(|(v, e)| v.powi(*e))
                .product()
        });

        Projection::Dense(activations.collect())
    }

    fn dim(&self) -> usize { self.limits.len() }

    fn size(&self) -> usize { self.exponents.len() }

    fn activity(&self) -> usize { self.size() }

    fn equivalent(&self, other: &Self) -> bool {
        self.order == other.order && self.limits == other.limits
    }
}
