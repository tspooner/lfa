use super::Transform;

// f(x) ≜ 1 / (1 + exp(-x))
pub struct Logistic;

impl Transform<f64> for Logistic {
    fn transform(&self, x: f64) -> f64 {
        1.0 / (1.0 + (-x).exp())
    }

    fn grad(&self, x: f64) -> f64 {
        let exp_x = x.exp();
        let exp_x_plus_1 = exp_x + 1.0;

        exp_x / exp_x_plus_1 / exp_x_plus_1
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use std::f64::consts::E;
    use super::{Logistic, Transform};

    #[test]
    fn test_f64() {
        assert!((Logistic.transform(0.0) - 0.5).abs() < 1e-7);
        assert!((Logistic.transform(1.0) - 1.0 / (1.0 + 1.0 / E)).abs() < 1e-7);
        assert!((Logistic.transform(2.0) - 1.0 / (1.0 + 1.0 / E / E)).abs() < 1e-7);

        assert!((Logistic.grad(0.0) - 0.25).abs() < 1e-5);
        assert!((Logistic.grad(1.0) - 0.196612).abs() < 1e-5);
        assert!((Logistic.grad(2.0) - 0.104994).abs() < 1e-5);
    }

    #[test]
    fn test_f64_positive() {
        fn prop_positive(x: f64) -> bool {
            Logistic.transform(x).is_sign_positive()
        }

        quickcheck(prop_positive as fn(f64) -> bool);
    }
}
