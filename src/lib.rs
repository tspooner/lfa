extern crate itertools;
extern crate ndarray;
extern crate rand;

pub extern crate spaces as geometry;

extern crate serde;
#[macro_use]
extern crate serde_derive;

mod utils;
mod error;
pub use self::error::*;

pub mod core;
pub use self::core::*;

pub mod approximators;
pub mod projectors;
pub use self::projectors::{AdaptiveProjector, Projection, Projector};

mod lfa;
pub use self::lfa::LFA;
