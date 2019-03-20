extern crate itertools;
extern crate ndarray;
extern crate rand;
extern crate serde;
#[macro_use]
extern crate serde_derive;

#[cfg(tests)]
extern crate quickcheck;

pub extern crate spaces as geometry;

mod macros;
mod utils;

import_all!(lfa);

#[macro_use]
pub mod core;
pub mod eval;
pub mod basis;
pub mod transforms;
