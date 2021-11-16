use super::*;
use crate::{ActivationT, Error, Features, IndexT};
use spaces::{BoundedSpace, Card, Dim, Interval, ProductSpace, Space};

mod cpfk;

// TODO: Implement this in a more computational efficient way.
//
// This is actually non-trivial for general multivariate polynomials. See
// https://eli.thegreenplace.net/2010/03/30/horners-rule-efficient-evaluation-of-polynomials
/// Polynomial basis projector.
///
/// ## Linear regression
/// ```
/// use lfa::basis::{Basis, Polynomial};
///
/// let p = Polynomial::new(1, 1);
///
/// assert_eq!(p.project(&vec![-1.0]).unwrap(), vec![-1.0].into());
/// assert_eq!(p.project(&vec![-0.5]).unwrap(), vec![-0.5].into());
/// assert_eq!(p.project(&vec![0.0]).unwrap(), vec![0.0].into());
/// assert_eq!(p.project(&vec![0.5]).unwrap(), vec![0.5].into());
/// assert_eq!(p.project(&vec![1.0]).unwrap(), vec![1.0].into());
/// ```
///
/// ## Quadratic regression
/// ```
/// use lfa::basis::{Basis, Polynomial};
///
/// let p = Polynomial::new(1, 2);
///
/// assert_eq!(p.project(&vec![-1.0]).unwrap(), vec![-1.0, 1.0].into());
/// assert_eq!(p.project(&vec![-0.5]).unwrap(), vec![-0.5, 0.25].into());
/// assert_eq!(p.project(&vec![0.0]).unwrap(), vec![0.0, 0.0].into());
/// assert_eq!(p.project(&vec![0.5]).unwrap(), vec![0.5, 0.25].into());
/// assert_eq!(p.project(&vec![1.00]).unwrap(), vec![1.0, 1.0].into());
/// ```
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Polynomial {
    pub order: u8,
    pub exponents: Vec<Vec<u8>>,
}

impl Polynomial {
    pub fn new(dim: usize, order: u8) -> Self {
        let exponents = compute_coefficients(order, dim).collect();

        Polynomial { order, exponents }
    }

    pub fn with_zeroth(mut self) -> Self {
        self.exponents.push(vec![0; self.exponents[0].len()]);
        self
    }

    fn compute_feature<F, I>(iter: I) -> f64
    where
        F: std::borrow::Borrow<f64>,
        I: IntoIterator<Item = (F, u8)>,
    {
        iter.into_iter()
            .map(|(v, e)| v.borrow().powi(e as i32))
            .product()
    }
}

impl Space for Polynomial {
    type Value = Features;

    fn dim(&self) -> Dim {
        Dim::Finite(self.exponents.len())
    }

    fn card(&self) -> Card {
        Card::Infinite
    }
}

impl<I: std::borrow::Borrow<f64>, T: IntoIterator<Item = I>> Basis<T> for Polynomial
where
    T::IntoIter: Clone,
{
    fn project(&self, input: T) -> crate::Result<Features> {
        let iter = input.into_iter().map(|x| *x.borrow());

        Ok(self
            .exponents
            .iter()
            .map(|exps| Self::compute_feature(iter.clone().zip(exps.iter().copied())))
            .collect())
    }
}

impl<I: std::borrow::Borrow<f64>, T: IntoIterator<Item = I>> EnumerableBasis<T> for Polynomial
where
    T::IntoIter: Clone,
{
    fn ith(&self, input: T, index: IndexT) -> crate::Result<ActivationT> {
        self.exponents
            .get(index)
            .map(|exps| {
                let iter = input
                    .into_iter()
                    .map(|x| *x.borrow())
                    .zip(exps.iter().copied());

                Self::compute_feature(iter)
            })
            .ok_or_else(|| Error::index_error(index, self.dim().into()))
    }
}

impl Combinators for Polynomial {}

/// Chebyshev polynomial basis projector.
#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize), serde(crate = "serde_crate"))]
pub struct Chebyshev {
    pub order: u8,
    pub limits: Vec<(f64, f64)>,

    #[cfg_attr(feature = "serde", serde(skip_serializing))]
    pub polynomials: Vec<Vec<fn(f64) -> f64>>,
}

impl Chebyshev {
    pub fn new(order: u8, limits: Vec<(f64, f64)>) -> Self {
        if order > 11 {
            panic!("Chebyshev only supports orders up to 11.")
        }

        let polynomials = Chebyshev::make_polynomials(order, limits.len());

        Chebyshev {
            order,
            limits,
            polynomials,
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

    pub fn with_zeroth(mut self) -> Self {
        self.polynomials.push(vec![|_| 1.0; self.limits.len()]);
        self
    }

    fn make_polynomials(order: u8, dim: usize) -> Vec<Vec<fn(f64) -> f64>> {
        let exponents = compute_coefficients(order, dim);

        exponents
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

    fn compute_feature<'a, I>(iter: I) -> f64
    where
        I: IntoIterator<Item = (f64, &'a fn(f64) -> f64)>,
    {
        iter.into_iter().map(|(v, f)| f(v)).product()
    }
}

impl Space for Chebyshev {
    type Value = Features;

    fn dim(&self) -> Dim {
        Dim::Finite(self.polynomials.len())
    }

    fn card(&self) -> Card {
        Card::Infinite
    }
}

impl<I: std::borrow::Borrow<f64>, T: IntoIterator<Item = I>> Basis<T> for Chebyshev
where
    T::IntoIter: Clone,
{
    fn project(&self, input: T) -> crate::Result<Features> {
        let scaled_state: Vec<f64> = rescale!(input into self.limits)
            .map(|v| 2.0 * v - 1.0)
            .collect();

        Ok(self
            .polynomials
            .iter()
            .map(|ps| Self::compute_feature(scaled_state.iter().copied().zip(ps.iter())))
            .collect())
    }
}

impl<I: std::borrow::Borrow<f64>, T: IntoIterator<Item = I>> EnumerableBasis<T> for Chebyshev
where
    T::IntoIter: Clone,
{
    fn ith(&self, input: T, index: IndexT) -> crate::Result<ActivationT> {
        self.polynomials
            .get(index)
            .map(|ps| {
                let ss = rescale!(input into self.limits).map(|v| 2.0 * v - 1.0);

                Self::compute_feature(ss.zip(ps.iter()))
            })
            .ok_or_else(|| Error::index_error(index, self.dim().into()))
    }
}

impl Combinators for Chebyshev {}

#[cfg(feature = "serde")]
use serde::{
    de::{self, MapAccess, SeqAccess, Visitor},
    Deserialize, Deserializer,
};
#[cfg(feature = "serde")]
use std::fmt;
#[cfg(feature = "serde")]
impl<'de> Deserialize<'de> for Chebyshev {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[cfg_attr(
            feature = "serde",
            derive(Deserialize),
            serde(crate = "serde_crate"),
            serde(field_identifier, rename_all = "lowercase")
        )]
        enum Field {
            Order,
            Limits,
        }

        struct ChebyshevVisitor;

        impl<'de> Visitor<'de> for ChebyshevVisitor {
            type Value = Chebyshev;

            fn expecting(&self, formatter: &mut fmt::Formatter) -> fmt::Result {
                formatter.write_str("struct Chebyshev")
            }

            fn visit_seq<V>(self, mut seq: V) -> std::result::Result<Chebyshev, V::Error>
            where
                V: SeqAccess<'de>,
            {
                let order = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(0, &self))?;
                let limits = seq
                    .next_element()?
                    .ok_or_else(|| de::Error::invalid_length(1, &self))?;

                Ok(Chebyshev::new(order, limits))
            }

            fn visit_map<V>(self, mut map: V) -> std::result::Result<Chebyshev, V::Error>
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
