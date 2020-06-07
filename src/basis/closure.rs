use super::*;
use crate::{Features, Result};
use spaces::{Card, Dim, Space};

#[derive(Clone, Debug, PartialEq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Closure<F> {
    n_features: usize,
    mapping: F,
}

impl<F> Closure<F> {
    pub fn new(n_features: usize, mapping: F) -> Self {
        Closure {
            n_features,
            mapping,
        }
    }
}

impl<F> Space for Closure<F> {
    type Value = Features;

    fn dim(&self) -> Dim { Dim::Finite(self.n_features) }

    fn card(&self) -> Card { Card::Infinite }
}

impl<T, F> Basis<T> for Closure<F>
where F: Fn(T) -> Result<Features>
{
    fn project(&self, input: T) -> Result<Features> { (self.mapping)(input) }
}

impl<F> Combinators for Closure<F> {}
