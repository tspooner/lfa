use crate::geometry::Vector;

pub trait ElementwiseProduct<T = Self> {
    type Output;

    fn elementwise_product(self, other: T) -> Self::Output;
}

// f64:
impl ElementwiseProduct for f64 {
    type Output = f64;

    fn elementwise_product(self, other: f64) -> f64 {
        self * other
    }
}

// [f64; 2]:
impl ElementwiseProduct for [f64; 2] {
    type Output = [f64; 2];

    fn elementwise_product(self, other: [f64; 2]) -> [f64; 2] {
        [self[0] * other[0], self[1] * other[1]]
    }
}

impl ElementwiseProduct<f64> for [f64; 2] {
    type Output = [f64; 2];

    fn elementwise_product(self, scalar: f64) -> [f64; 2] {
        [self[0] * scalar, self[1] * scalar]
    }
}

// [f64; 3]:
impl ElementwiseProduct for [f64; 3] {
    type Output = [f64; 3];

    fn elementwise_product(self, other: [f64; 3]) -> [f64; 3] {
        [self[0] * other[0], self[1] * other[1], self[2] * other[2]]
    }
}

impl ElementwiseProduct<f64> for [f64; 3] {
    type Output = [f64; 3];

    fn elementwise_product(self, scalar: f64) -> [f64; 3] {
        [self[0] * scalar, self[1] * scalar, self[2] * scalar]
    }
}

// Vector<f64>:
impl ElementwiseProduct for Vector<f64> {
    type Output = Vector<f64>;

    fn elementwise_product(self, other: Self) -> Self {
        self * other
    }
}

impl ElementwiseProduct<f64> for Vector<f64> {
    type Output = Vector<f64>;

    fn elementwise_product(self, scalar: f64) -> Vector<f64> {
        self * scalar
    }
}
