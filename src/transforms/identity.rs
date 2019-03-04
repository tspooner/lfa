use geometry::Vector;
use super::Transform;

// f(x) ≜ x
pub struct Identity;

macro_rules! impl_identity {
    ($type:ty; $grad:expr) => {
        impl Transform<$type> for Identity {
            fn transform(&self, x: $type) -> $type { x }

            fn grad(&self, _: $type) -> $type { $grad }
        }
    }
}

impl_identity!(f64; 1.0);
impl_identity!((f64, f64); (1.0, 1.0));
impl_identity!((f64, f64, f64); (1.0, 1.0, 1.0));

impl Transform<Vector<f64>> for Identity {
    fn transform(&self, x: Vector<f64>) -> Vector<f64> { x }

    fn grad(&self, mut x: Vector<f64>) -> Vector<f64> {
        x.fill(1.0);
        x
    }
}

#[cfg(test)]
mod tests {
    use super::{Identity, Transform};

    #[test]
    fn test_f64() {
        assert_eq!(Identity.transform(0.0), 0.0);
        assert_eq!(Identity.transform(1.0), 1.0);
        assert_eq!(Identity.transform(2.0), 2.0);

        assert_eq!(Identity.grad(0.0), 1.0);
        assert_eq!(Identity.grad(1.0), 1.0);
        assert_eq!(Identity.grad(2.0), 1.0);
    }
}
