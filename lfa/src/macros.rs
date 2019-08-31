#![macro_use]

macro_rules! import_all {
    ($module:ident) => {
        mod $module;
        pub use self::$module::*;
    };
    ($module:ident with macros) => {
        #[macro_use]
        mod $module;
        pub use self::$module::*;
    };
}

#[cfg(test)]
macro_rules! assert_features {
    ($projector:ident +- $tol:literal [$($input:expr => $output:expr),+]) => {{
        $(assert!($crate::utils::compare_floats(
            $projector.project(&$input).unwrap().expanded().into_raw_vec(), $output, $tol,
        ));)+
    }}
}
