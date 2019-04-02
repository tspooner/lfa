use crate::{
    basis::Composable,
    core::{Projection, Projector},
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

mod cpfk;

/// Polynomial basis projector.
///
/// ## Linear regression on the interval [-1, 1]
/// ```
/// use lfa::basis::{Projector, fixed::Polynomial};
///
/// let p = Polynomial::new(1, vec![(0.0, 1.0)]);
///
/// assert_eq!(p.project(&vec![0.00]), vec![-1.0].into());
/// assert_eq!(p.project(&vec![0.25]), vec![-0.5].into());
/// assert_eq!(p.project(&vec![0.50]), vec![0.0].into());
/// assert_eq!(p.project(&vec![0.75]), vec![0.5].into());
/// assert_eq!(p.project(&vec![1.00]), vec![1.0].into());
/// ```
///
/// ## Quadratic regression on the interval [-1, 1]
/// ```
/// use lfa::basis::{Projector, fixed::Polynomial};
///
/// let p = Polynomial::new(2, vec![(0.0, 1.0)]);
///
/// assert_eq!(p.project(&vec![0.00]), vec![-1.0, 1.0].into());
/// assert_eq!(p.project(&vec![0.25]), vec![-0.5, 0.25].into());
/// assert_eq!(p.project(&vec![0.50]), vec![0.0, 0.0].into());
/// assert_eq!(p.project(&vec![0.75]), vec![0.5, 0.25].into());
/// assert_eq!(p.project(&vec![1.00]), vec![1.0, 1.0].into());
/// ```
#[derive(Clone, Serialize, Deserialize, Debug)]
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

    pub fn from_space(order: u8, input_space: LinearSpace<Interval>) -> Self {
        Polynomial::new(
            order,
            input_space
                .iter()
                .map(|d| (d.inf().unwrap(), d.sup().unwrap()))
                .collect(),
        )
    }

    fn make_exponents(order: u8, dim: usize) -> Vec<Vec<i32>> {
        let dcs = vec![(0..(order + 1)).map(|v| v as i32).collect::<Vec<i32>>(); dim];
        let mut exponents = cartesian_product(&dcs);

        exponents.sort_by(|a, b| b.partial_cmp(a).unwrap());
        exponents.dedup();
        exponents.pop();
        exponents.reverse();

        exponents
    }
}

impl Space for Polynomial {
    type Value = Projection;

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

        Projection::Dense(
            self.exponents
                .iter()
                .map(|exps| {
                    scaled_state
                        .iter()
                        .zip(exps)
                        .map(|(v, e)| v.powi(*e))
                        .product()
                })
                .collect(),
        )
    }
}

impl_array_proxies!(Polynomial; f64);

impl Composable for Polynomial {}

/// Chebyshev polynomial basis projector.
#[derive(Clone, Debug)]
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

    pub fn from_space(order: u8, input_space: LinearSpace<Interval>) -> Self {
        Chebyshev::new(
            order,
            input_space
                .iter()
                .map(|d| (d.inf().unwrap(), d.sup().unwrap()))
                .collect(),
        )
    }

    fn make_polynomials(order: u8, dim: usize) -> Vec<Vec<fn(f64) -> f64>> {
        let dcs = vec![(0..(order + 1)).collect::<Vec<u8>>(); dim];
        let mut coefficients = cartesian_product(&dcs);

        coefficients.sort_by(|a, b| a.partial_cmp(b).unwrap());
        coefficients.dedup();

        coefficients
            .iter()
            .skip(1)
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

        Projection::Dense(
            self.polynomials
                .iter()
                .map(|ps| scaled_state.iter().zip(ps).map(|(v, f)| f(*v)).product())
                .collect(),
        )
    }
}

impl_array_proxies!(Chebyshev; f64);

impl Composable for Chebyshev {}
