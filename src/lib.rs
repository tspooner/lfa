extern crate itertools;
extern crate ndarray;
extern crate rand;

pub extern crate spaces as geometry;

extern crate serde;
#[macro_use]
extern crate serde_derive;

mod macros;
mod utils;

import_all!(core);

mod approximators;

pub mod models;
pub mod projectors;
pub use self::projectors::{AdaptiveProjector, Projection, Projector};
