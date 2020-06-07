//! LFA error and result types.
use crate::IndexT;
use std::{error::Error as StdError, fmt};

pub type Result<T> = ::std::result::Result<T, Error>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub enum ErrorKind {
    Basis,
    Evaluation,
    Optimisation,
}

#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(
    feature = "serde",
    derive(Serialize, Deserialize),
    serde(crate = "serde_crate")
)]
pub struct Error {
    kind: ErrorKind,
    message: String,
}

impl Error {
    pub fn index_error(index: IndexT, dim: IndexT) -> Self {
        Error {
            kind: ErrorKind::Basis,
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

macro_rules! check_index {
    ($index:ident < $dim:expr => $code:block) => {
        if $index < $dim {
            $code
        } else {
            Err($crate::error::Error::index_error($index, $dim))
        }
    };
}
