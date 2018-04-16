use geometry::{BoundedSpace, RegularSpace, Space, Card, Vector, dimensions::Continuous};
use rand::ThreadRng;
use super::{Projection, Projector};
use utils::cartesian_product;

mod cpfk;

/// Polynomial basis projector.
#[derive(Clone, Serialize, Deserialize)]
pub struct Polynomial {
    pub order: u8,
    pub limits: Vec<(f64, f64)>,
    pub exponents: Vec<Vec<i32>>,
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

    pub fn from_space(order: u8, input_space: RegularSpace<Continuous>) -> Self {
        Polynomial::new(
            order,
            input_space.iter().map(|d| (*d.lb(), *d.ub())).collect(),
        )
    }

    fn make_exponents(order: u8, dim: usize) -> Vec<Vec<i32>> {
        let dcs = vec![(0..(order + 1)).map(|v| v as i32).collect::<Vec<i32>>(); dim];
        let mut exponents = cartesian_product(&dcs);

        exponents.sort_by(|a, b| b.partial_cmp(a).unwrap());
        exponents.dedup();

        exponents
    }
}

impl Space for Polynomial {
    type Value = Projection;

    fn sample(&self, _rng: &mut ThreadRng) -> Projection { unimplemented!() }

    fn dim(&self) -> usize { self.exponents.len() }

    fn card(&self) -> Card { Card::Infinite }
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
}

/// Chebyshev polynomial basis projector.
#[derive(Clone)]
pub struct Chebyshev {
    pub order: u8,
    pub limits: Vec<(f64, f64)>,
    pub polynomials: Vec<Vec<fn(f64) -> f64>>,
}

impl Chebyshev {
    pub fn new(order: u8, limits: Vec<(f64, f64)>) -> Self {
        if order > 11 {
            panic!("Chebyshev only supports orders up to 11.")
        }

        let polynomials = Chebyshev::make_polynomials(order, limits.len());

        Chebyshev {
            order: order,
            limits: limits,
            polynomials: polynomials,
        }
    }

    pub fn from_space(order: u8, input_space: RegularSpace<Continuous>) -> Self {
        Chebyshev::new(
            order,
            input_space.iter().map(|d| (*d.lb(), *d.ub())).collect(),
        )
    }

    fn make_polynomials(order: u8, dim: usize) -> Vec<Vec<fn(f64) -> f64>> {
        let dcs = vec![(0..(order + 1)).collect::<Vec<u8>>(); dim];
        let mut coefficients = cartesian_product(&dcs);

        coefficients.sort_by(|a, b| b.partial_cmp(a).unwrap());
        coefficients.dedup();

        coefficients
            .iter()
            .map(|vals| {
                vals.iter()
                    .map(|i| match *i {
                        0 => cpfk::t_0,
                        1 => cpfk::t_1,
                        2 => cpfk::t_2,
                        3 => cpfk::t_3,
                        4 => cpfk::t_4,
                        5 => cpfk::t_5,
                        6 => cpfk::t_6,
                        7 => cpfk::t_7,
                        8 => cpfk::t_8,
                        9 => cpfk::t_9,
                        10 => cpfk::t_10,
                        11 => cpfk::t_11,
                        _ => panic!("Chebyshev only supports orders up to 11."),
                    })
                    .collect()
            })
            .collect()
    }
}

impl Space for Chebyshev {
    type Value = Projection;

    fn sample(&self, _rng: &mut ThreadRng) -> Projection { unimplemented!() }

    fn dim(&self) -> usize { self.polynomials.len() }

    fn card(&self) -> Card { Card::Infinite }
}

impl Projector<[f64]> for Chebyshev {
    fn project(&self, input: &[f64]) -> Projection {
        let scaled_state = input
            .iter()
            .enumerate()
            .map(|(i, v)| (v - self.limits[i].0) / (self.limits[i].1 - self.limits[i].0))
            .map(|v| 2.0 * v - 1.0)
            .collect::<Vec<f64>>();

        let activations = self.polynomials
            .iter()
            .map(|polys| scaled_state.iter().zip(polys).map(|(v, t)| t(*v)).product());

        Projection::Dense(Vector::from_iter(activations))
    }
}
