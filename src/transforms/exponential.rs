use super::Transform;

// f(x) ≜ exp(x)
pub struct Exp;

impl Transform<f64> for Exp {
    fn transform(&self, x: f64) -> f64 {
        x.exp()
    }

    fn grad(&self, x: f64) -> f64 {
        self.transform(x)
    }
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;
    use std::f64::consts::E;
    use super::{Exp, Transform};

    #[test]
    fn test_f64() {
        assert!((Exp.transform(0.0) - 1.0).abs() < 1e-7);
        assert!((Exp.transform(1.0) - E).abs() < 1e-7);
        assert!((Exp.transform(2.0) - E * E).abs() < 1e-7);

        assert!((Exp.transform(0.0) - Exp.grad(0.0)).abs() < 1e-7);
        assert!((Exp.transform(1.0) - Exp.grad(1.0)).abs() < 1e-7);
        assert!((Exp.transform(2.0) - Exp.grad(2.0)).abs() < 1e-7);
    }

    #[test]
    fn test_f64_positive() {
        fn prop_positive(x: f64) -> bool {
            Exp.transform(x).is_sign_positive()
        }

        quickcheck(prop_positive as fn(f64) -> bool);
    }
}
