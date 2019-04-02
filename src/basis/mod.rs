//! Module for basis function representations used by `LFA`.
import_all!(composable);

pub mod composition;
pub mod fixed;
pub mod ifdd;

pub use crate::{
    core::{Projector, Features},
    geometry::kernels,
};
