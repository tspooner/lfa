use crate::geometry::Vector;
use super::Transform;

// f(x) ≜ exp(x)
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
#[derive(Copy, Clone, Debug)]
pub struct Exp;

impl Transform<f64> for Exp {
    type Output = f64;

    fn transform(&self, x: f64) -> f64 {
        x.exp()
    }

    fn grad(&self, x: f64) -> f64 {
        self.transform(x)
    }
}

impl Transform<[f64; 2]> for Exp {
    type Output = [f64; 2];

    fn transform(&self, x: [f64; 2]) -> [f64; 2] {
        [x[0].exp(), x[1].exp()]
    }

    fn grad(&self, x: [f64; 2]) -> [f64; 2] {
        self.transform(x)
    }
}

impl Transform<[f64; 3]> for Exp {
    type Output = [f64; 3];

    fn transform(&self, x: [f64; 3]) -> [f64; 3] {
        [x[0].exp(), x[1].exp(), x[2].exp()]
    }

    fn grad(&self, x: [f64; 3]) -> [f64; 3] {
        self.transform(x)
    }
}

impl Transform<Vector<f64>> for Exp {
    type Output = Vector<f64>;

    fn transform(&self, x: Vector<f64>) -> Vector<f64> {
        x.mapv_into(|v| v.exp())
    }

    fn grad(&self, x: Vector<f64>) -> Vector<f64> {
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
