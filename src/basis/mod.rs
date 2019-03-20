//! Module for basis function representations used by `LFA`.
import_all!(composable);

pub mod adaptive;
pub mod composition;
pub mod fixed;

pub use crate::{
    core::{Projector, AdaptiveProjector, Projection, Feature, CandidateFeature},
    geometry::kernels,
};
