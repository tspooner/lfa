use crate::{
    composition::Composable,
    core::{DenseT, Features},
    geometry::Space,
};

/// Trait for basis projectors.
pub trait Projector<I: ?Sized>: Space<Value = Features> {
    /// Project data from an input space onto the basis.
    ///
    /// ```
    /// use lfa::basis::{Projector, Features, fixed::Constant};
    ///
    /// let projector = Constant::ones(2);
    ///
    /// assert!(projector.project(&[0.0]).is_dense());
    /// assert_eq!(
    ///     projector.project(&[0.0]),
    ///     Features::from(vec![1.0, 1.0])
    /// );
    /// ```
    fn project(&self, input: &I) -> Features;

    /// Project data from an input space onto the basis and expand into a dense
    /// vector form using `Features::expanded(self.dim())`.
    ///
    /// ```
    /// use lfa::basis::{Projector, Features, fixed::Indices};
    ///
    /// let projector = Indices::new(2, vec![0, 1]);
    ///
    /// assert!(projector.project(&[0.0]).is_sparse());
    /// assert_eq!(
    ///     projector.project(&[0.0]),
    ///     Features::from(vec![0, 1]),
    /// );
    /// assert_eq!(
    ///     projector.project_expanded(&[0.0]),
    ///     vec![1.0, 1.0].into(),
    /// );
    /// ```
    fn project_expanded(&self, input: &I) -> DenseT { self.project(input).expanded(self.dim()) }
}

// #[macro_export]
macro_rules! impl_array_proxy {
    ([$itype:ty; $($n:expr),*] for $type:ty) => {
        $(
            impl Projector<[$itype; $n]> for $type where $type: Projector<[$itype]> {
                fn project(&self, input: &[$itype; $n]) -> Features {
                    Projector::<[$itype]>::project(self, input)
                }
            }
        )*
    };
    ([$itype:ty; +] for $type:ty) => {
        impl_array_proxy!([$itype;
             1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24
        ] for $type);
    };
    (Vec<$itype:ty> for $type:ty) => {
        impl Projector<Vec<$itype>> for $type where $type: Projector<[$itype]> {
            fn project(&self, input: &Vec<$itype>) -> Features {
                Projector::<[$itype]>::project(self, &input)
            }
        }
    };
    (Vector<$itype:ty> for $type:ty) => {
        impl Projector<Vector<$itype>> for $type where $type: Projector<[$itype]> {
            fn project(&self, input: &Vector<$itype>) -> Features {
                Projector::<[$itype]>::project(self, input.as_slice().unwrap())
            }
        }
    };
}

// #[macro_export]
macro_rules! impl_array_proxies {
    ($type:ty; $($itype:ty),*) => {
        $(
            // impl_array_proxy!([$itype; +] for $type);
            impl_array_proxy!(Vec<$itype> for $type);
            impl_array_proxy!(Vector<$itype> for $type);
        )*
    }
}
