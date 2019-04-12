//! Module for basis function representations used by `LFA`.
pub use crate::{
    core::Features,
    geometry::kernels,
};

import_all!(projector with macros);

pub mod fixed;
pub mod ifdd;
