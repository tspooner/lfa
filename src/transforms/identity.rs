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
