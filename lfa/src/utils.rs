#![allow(dead_code)]

pub(self) fn partial_cartesian<T: Clone>(a: Vec<Vec<T>>, b: &Vec<T>) -> Vec<Vec<T>> {
    a.into_iter()
        .flat_map(|xs| {
            b.iter()
                .cloned()
                .map(|y| {
                    let mut vec = xs.clone();
                    vec.push(y);
                    vec
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

pub(crate) fn cartesian_product<T: Clone>(lists: &Vec<Vec<T>>) -> Vec<Vec<T>> {
    match lists.split_first() {
        Some((first, rest)) => {
            let init: Vec<Vec<T>> = first.iter().cloned().map(|n| vec![n]).collect();

            rest.iter()
                .cloned()
                .fold(init, |vec, list| partial_cartesian(vec, &list))
        },
        None => vec![],
    }
}

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
    a.into_iter().zip(b.into_iter()).all(move |(x, y)| (x.borrow() - y.borrow()).abs() < tol)
}
