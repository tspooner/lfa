extern crate rand;
extern crate ndarray;
extern crate itertools;

pub extern crate spaces as geometry;

extern crate serde;
#[macro_use]
extern crate serde_derive;

mod utils;

mod error;
pub use self::error::*;

pub mod projectors;
pub use self::projectors::{Projection, Projector, AdaptiveProjector};

pub mod approximators;
pub use self::approximators::Approximator;

mod lfa;
pub use self::lfa::LFA;
