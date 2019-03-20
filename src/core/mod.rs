//! Module for common types and primitives used throughout the crate.
mod calculus;
pub(crate) use self::calculus::Gradient;

import_all!(primitives);
import_all!(error);

import_all!(approximator);

import_all!(feature);
import_all!(projection);
import_all!(projector with macros);

import_all!(parameterised);
