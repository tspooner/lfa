use crate::{
    IndexT, ActivationT, Features,
    basis::Projector,
    utils::cartesian_product,
};
use spaces::{
    BoundedSpace,
    Interval, ProductSpace,
};

mod cpfk;

/// Polynomial basis projector.
///
/// ## Linear regression
/// ```
/// use lfa::basis::{Projector, Polynomial};
///
/// let p = Polynomial::new(1, 1);
///
/// assert_eq!(p.project(&vec![-1.0]), vec![-1.0].into());
/// assert_eq!(p.project(&vec![-0.5]), vec![-0.5].into());
/// assert_eq!(p.project(&vec![0.0]), vec![0.0].into());
/// assert_eq!(p.project(&vec![0.5]), vec![0.5].into());
/// assert_eq!(p.project(&vec![1.0]), vec![1.0].into());
/// ```
///
/// ## Quadratic regression
/// ```
/// use lfa::basis::{Projector, Polynomial};
///
/// let p = Polynomial::new(1, 2);
///
/// assert_eq!(p.project(&vec![-1.0]), vec![-1.0, 1.0].into());
/// assert_eq!(p.project(&vec![-0.5]), vec![-0.5, 0.25].into());
/// assert_eq!(p.project(&vec![0.0]), vec![0.0, 0.0].into());
/// assert_eq!(p.project(&vec![0.5]), vec![0.5, 0.25].into());
/// assert_eq!(p.project(&vec![1.00]), vec![1.0, 1.0].into());
/// ```
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Polynomial {
    pub order: u8,
    pub exponents: Vec<Vec<i32>>,
}

impl Polynomial {
    pub fn new(dim: usize, order: u8) -> Self {
        let exponents = Polynomial::make_exponents(dim, order);

        Polynomial {
            order: order,
            exponents: exponents,
        }
    }

    fn make_exponents(dim: usize, order: u8) -> Vec<Vec<i32>> {
        let dcs = vec![(0..=order).map(|v| v as i32).collect::<Vec<i32>>(); dim];
        let mut exponents: Vec<Vec<i32>> = cartesian_product(&dcs)
            .into_iter()
            .filter(|exps| exps.iter().fold(0, |acc, e| acc + e) <= order as i32)
            .collect();

        exponents.sort_by(|a, b| b.partial_cmp(a).unwrap());
        exponents.dedup();
        exponents.pop();
        exponents.reverse();

        exponents
    }

    fn compute_feature(&self, ss: &[f64], exps: &[i32]) -> f64 {
        ss.iter().zip(exps).map(|(v, e)| v.powi(*e)).product()
    }
}

impl Projector for Polynomial {
    fn n_features(&self) -> usize { self.exponents.len() }

    fn project_ith(&self, input: &[f64], i: IndexT) -> Option<ActivationT> {
        Some(self.compute_feature(input, &self.exponents[i]))
    }

    fn project(&self, input: &[f64]) -> Features {
        self.exponents
            .iter()
            .map(|exps| self.compute_feature(input, exps))
            .collect()
    }
}

/// Chebyshev polynomial basis projector.
#[derive(Clone, Debug)]
#[cfg_attr(feature = "serialize", derive(Serialize))]
pub struct Chebyshev {
    pub order: u8,
    pub limits: Vec<(f64, f64)>,

    #[cfg_attr(feature = "serialize", serde(skip_serializing))]
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

    pub fn from_space(order: u8, input_space: ProductSpace<Interval>) -> Self {
        Chebyshev::new(
            order,
            input_space
                .iter()
                .map(|d| (d.inf().unwrap(), d.sup().unwrap()))
                .collect(),
        )
    }

    fn make_polynomials(order: u8, dim: usize) -> Vec<Vec<fn(f64) -> f64>> {
        let dcs = vec![(0..=order).collect::<Vec<u8>>(); dim];
        let mut coefficients: Vec<Vec<u8>> = cartesian_product(&dcs)
            .into_iter()
            .filter(|cs| cs.iter().fold(0, |acc, c| acc + c) <= order)
            .collect();

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

    fn rescale_input(&self, input: &[f64]) -> Vec<f64> {
        input
            .iter()
            .enumerate()
            .map(|(i, v)| {
                let v = (v - self.limits[i].0) / (self.limits[i].1 - self.limits[i].0);

                2.0 * v - 1.0
            })
            .collect::<Vec<f64>>()
    }

    fn compute_feature(&self, ss: &[f64], ps: &[fn(f64) -> f64]) -> f64 {
        ss.iter().zip(ps).map(|(v, f)| f(*v)).product()
    }
}

impl Projector for Chebyshev {
    fn n_features(&self) -> usize { self.polynomials.len() }

    fn project_ith(&self, input: &[f64], i: IndexT) -> Option<ActivationT> {
        Some(self.compute_feature(&self.rescale_input(input), &self.polynomials[i]))
    }

    fn project(&self, input: &[f64]) -> Features {
        let scaled_state = self.rescale_input(input);

        self.polynomials
            .iter()
            .map(|ps| self.compute_feature(&scaled_state, ps))
            .collect()
    }
}

#[cfg(feature = "serialize")] use serde::{
    Deserialize,
    Deserializer,
    de::{self, Visitor, SeqAccess, MapAccess}
};
#[cfg(feature = "serialize")] use std::fmt;
#[cfg(feature = "serialize")]
impl<'de> Deserialize<'de> for Chebyshev {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        #[serde(field_identifier, rename_all = "lowercase")]
        enum Field { Order, Limits }

        struct ChebyshevVisitor;

        impl<'de> Visitor<'de> for ChebyshevVisitor {
            type Value = Chebyshev;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Chebyshev")
            }

            fn visit_seq<V>(self, mut seq: V) -> Result<Chebyshev, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let order = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let limits = seq.next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;

                Ok(Chebyshev::new(order, limits))
            }

            fn visit_map<V>(self, mut map: V) -> Result<Chebyshev, V::Error>
            where
                V: MapAccess<'de>,
            {
                let mut order = None;
                let mut limits = None;
                while let Some(key) = map.next_key()? {
                    match key {
                        Field::Order => {
                            if order.is_some() {
                                return Err(de::Error::duplicate_field("order"));
                            }
                            order = Some(map.next_value()?);
                        }
                        Field::Limits => {
                            if limits.is_some() {
                                return Err(de::Error::duplicate_field("limits"));
                            }
                            limits = Some(map.next_value()?);
                        }
                    }
                }

                let order = order.ok_or_else(|| de::Error::missing_field("order"))?;
                let limits = limits.ok_or_else(|| de::Error::missing_field("limits"))?;

                Ok(Chebyshev::new(order, limits))
            }
        }

        const FIELDS: &'static [&'static str] = &["order", "limits"];
        deserializer.deserialize_struct("Chebyshev", FIELDS, ChebyshevVisitor)
    }
}
