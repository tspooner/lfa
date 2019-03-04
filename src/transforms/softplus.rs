use super::{Transform, Logistic};

// f(x) ≜ log(1 + exp(x))
pub struct Softplus;

impl Transform<f64> for Softplus {
    fn transform(&self, x: f64) -> f64 {
        (1.0 + x.exp()).ln()
    }

    fn grad(&self, x: f64) -> f64 {
        Logistic.transform(x)
    }
}

#[cfg(test)]
mod tests {
    use crate::transforms::Logistic;
    use quickcheck::quickcheck;
    use super::{Softplus, Transform};

    #[test]
    fn test_f64() {
        assert!((Softplus.transform(0.0) - 0.693147).abs() < 1e-5);
        assert!((Softplus.transform(1.0) - 1.31326).abs() < 1e-5);
        assert!((Softplus.transform(2.0) - 2.12693).abs() < 1e-5);

        assert!((Softplus.grad(0.0) - Logistic.transform(0.0)).abs() < 1e-7);
        assert!((Softplus.grad(1.0) - Logistic.transform(1.0)).abs() < 1e-7);
        assert!((Softplus.grad(2.0) - Logistic.transform(2.0)).abs() < 1e-7);
    }

    #[test]
    fn test_f64_positive() {
        fn prop_positive(x: f64) -> bool {
            Softplus.transform(x).is_sign_positive()
        }

        quickcheck(prop_positive as fn(f64) -> bool);
    }
}
