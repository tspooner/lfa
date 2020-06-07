#![macro_use]

#[cfg(test)]
macro_rules! assert_features {
    ($projector:ident +- $tol:literal [$($input:expr => $output:expr),+]) => {{
        $(assert!($crate::utils::compare_floats(
            $projector.project(&$input).unwrap().into_dense().into_raw_vec(), $output, $tol,
        ));)+
    }}
}
