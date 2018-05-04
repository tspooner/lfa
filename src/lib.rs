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

pub mod projectors;
pub use self::projectors::{AdaptiveProjector, Projection, Projector};

pub mod approximators;
pub use self::approximators::Approximator;

mod lfa;
pub use self::lfa::LFA;
