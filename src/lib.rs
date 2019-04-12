#[macro_use]
extern crate ndarray;
extern crate rand;
extern crate itertools;

extern crate serde;
#[macro_use]
extern crate serde_derive;

#[cfg(tests)]
extern crate quickcheck;

pub extern crate spaces as geometry;

mod macros;
mod utils;

#[macro_use]
pub mod core;
pub mod eval;
#[macro_use]
pub mod basis;
pub mod transforms;
pub mod composition;

import_all!(lfa);
