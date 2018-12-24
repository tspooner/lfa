use crate::core::{Projector, Projection};
use crate::geometry::{Card, Space};

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_project() {
        assert_eq!(Constant::zeros(1).project(&[0.0]), vec![0.0; 1].into());
        assert_eq!(Constant::zeros(10).project(&[0.0]), vec![0.0; 10].into());
        assert_eq!(Constant::zeros(100).project(&[0.0]), vec![0.0; 100].into());

        assert_eq!(Constant::ones(1).project(&[0.0]), vec![1.0; 1].into());
        assert_eq!(Constant::ones(10).project(&[0.0]), vec![1.0; 10].into());
        assert_eq!(Constant::ones(100).project(&[0.0]), vec![1.0; 100].into());

        assert_eq!(Constant::new(10, -1.5).project(&[0.0]), vec![-1.5; 10].into());
        assert_eq!(Constant::new(10, 5.6).project(&[0.0]), vec![5.6; 10].into());
        assert_eq!(Constant::new(10, 123.0).project(&[0.0]), vec![123.0; 10].into());
    }
}
