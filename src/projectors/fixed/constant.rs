use geometry::{Card, Space};
use projectors::{Projection, Projector};
use rand::{ThreadRng};

/// Fixed uniform basis projector.
#[derive(Clone)]
pub struct Constant {
    n_features: usize,
    value: f64,
}

impl Constant {
    pub fn new(n_features: usize, value: f64) -> Self {
        Constant {
            n_features: n_features,
            value: value,
        }
    }

    pub fn zeros(n_features: usize) -> Self {
        Constant::new(n_features, 0.0)
    }

    pub fn ones(n_features: usize) -> Self {
        Constant::new(n_features, 1.0)
    }
}

impl Space for Constant {
    type Value = Projection;

    fn sample(&self, _: &mut ThreadRng) -> Projection {
        vec![self.value; self.n_features].into()
    }

    fn dim(&self) -> usize {
        self.n_features
    }

    fn card(&self) -> Card {
        unimplemented!()
    }
}

impl Projector<[f64]> for Constant {
    fn project(&self, _: &[f64]) -> Projection {
        vec![self.value; self.n_features].into()
    }
}
