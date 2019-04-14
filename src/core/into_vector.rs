use geometry::Vector;

pub trait IntoVector {
    fn into_vector(self) -> Vector<f64>;
}

impl IntoVector for f64 {
    fn into_vector(self) -> Vector<f64> { Vector::from_elem(1, self) }
}

impl IntoVector for [f64; 2] {
    fn into_vector(self) -> Vector<f64> { Vector::from_vec(vec![self[0], self[1]]) }
}

impl IntoVector for [f64; 3] {
    fn into_vector(self) -> Vector<f64> { Vector::from_vec(vec![self[0], self[1], self[2]]) }
}

impl IntoVector for Vector<f64> {
    fn into_vector(self) -> Vector<f64> { self }
}
