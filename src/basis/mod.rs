//! Module for basis function representations used by `LFA`.
import_all!(feature);
import_all!(projection);
import_all!(projector with macros);
import_all!(composable);

pub mod adaptive;
pub mod composition;
pub mod fixed;

pub use crate::geometry::kernels;
