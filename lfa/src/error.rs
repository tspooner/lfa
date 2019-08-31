//! LFA error and result types.
use crate::IndexT;
use std::{error::Error as StdError, fmt};

pub type Result<T> = ::std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub enum ErrorKind {
    Evaluation,
    Projection,
    Optimisation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    pub fn index_error(index: IndexT, dim: IndexT) -> Self {
        Error {
            kind: ErrorKind::Projection,
            message: format!(
                "Index ({}) exceeded dimensionality ({}) of the projection.",
                index, dim
            ),
        }
    }
}

impl fmt::Display for Error {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result { writeln!(f, "{}", self.message) }
}

impl StdError for Error {
    fn description(&self) -> &str { &*self.message }
}

#[inline(always)]
pub(crate) fn check_index<T>(index: IndexT, dim: IndexT, f: impl Fn() -> Result<T>) -> Result<T> {
    if index < dim {
        f()
    } else {
        Err(Error::index_error(index, dim))
    }
}
