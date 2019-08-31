#![allow(dead_code)]

pub(crate) fn eq_with_nan_eq(a: f64, b: f64, tol: f64) -> bool {
    (a.is_nan() && b.is_nan()) || (a - b).abs() < tol
}

pub(crate) fn compare_floats<T1, I1, T2, I2>(a: I1, b: I2, tol: f64) -> bool
where
    T1: std::borrow::Borrow<f64>,
    I1: IntoIterator<Item = T1>,
    T2: std::borrow::Borrow<f64>,
    I2: IntoIterator<Item = T2>,
{
    a.into_iter()
        .zip(b.into_iter())
        .all(move |(x, y)| (x.borrow() - y.borrow()).abs() < tol)
}
