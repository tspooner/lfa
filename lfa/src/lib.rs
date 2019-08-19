//! # LFA
//!
//! LFA is a framework for linear function approximation with gradient descent.
#[allow(unused_imports)]
#[macro_use]
extern crate lfa_derive;
#[doc(hidden)]
pub use self::lfa_derive::*;

extern crate itertools;
extern crate elementwise;

#[cfg(feature = "serialize")] extern crate serde;
#[cfg_attr(feature = "serialize", macro_use)]
#[cfg(feature = "serialize")]
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

