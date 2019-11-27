use crate::{IndexT, ActivationT, Features, Result, Error, basis::Basis};

/// Stack the output of two `Basis` instances.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Stacker<B1, B2> {
    b1: B1,
    b2: B2,
}

impl<B1, B2> Stacker<B1, B2> {
    pub fn new(b1: B1, b2: B2) -> Self { Stacker { b1, b2 } }
}

impl<B1, B2> Basis for Stacker<B1, B2>
where
    B1: Basis,
    B2: Basis,
{
    fn n_features(&self) -> usize {
        self.b1.n_features() + self.b2.n_features()
    }

    fn project_ith(&self, input: &[f64], index: IndexT) -> Result<Option<ActivationT>> {
        let n1 = self.b1.n_features();
        let n2 = self.b2.n_features();
        let n12 = n1 + n2;

        if index < n1 {
            self.b1.project_ith(input, index)
        } else if index < n12 {
            self.b2.project_ith(input, index - n1)
        } else {
            Err(Error::index_error(index, n12))
        }
    }

    fn project(&self, input: &[f64]) -> Result<Features> {
        self.b1.project(input).and_then(|f1| {
            self.b2.project(input).map(|f2| f1.stack(f2))
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::basis::{Basis, Constants, Indices};
    use std::iter;
    use super::*;

    #[test]
    fn test_stack_constant() {
        let p = Stacker::new(Constants::new(vec![0.0; 10]), Constants::new(vec![1.0; 10]));
        let output = Features::Dense(
            iter::repeat(0.0).take(10)
                .chain(iter::repeat(1.0).take(10))
                .collect()
        );

        assert_eq!(p.n_features(), 20);

        assert_eq!(p.project(&[0.0]).unwrap(), output);
        assert_eq!(p.project(&[0.0, 1.0]).unwrap(), output);
        assert_eq!(p.project(&[-1.0, 1.0]).unwrap(), output);
    }

    #[test]
    fn test_stack_indices() {
        let p = Stacker::new(Indices::new(10, vec![5]), Indices::new(10, vec![0]));
        let output = Features::Sparse(20, vec![(5, 1.0), (10, 1.0)].into_iter().collect());

        assert_eq!(p.n_features(), 20);

        assert_eq!(p.project(&[0.0]).unwrap(), output);
        assert_eq!(p.project(&[0.0, 1.0]).unwrap(), output);
        assert_eq!(p.project(&[-1.0, 1.0]).unwrap(), output);
    }

    #[test]
    fn test_stack_mixed() {
        let p = Stacker::new(Constants::new(vec![1.0; 10]), Indices::new(10, vec![0]));
        let output = Features::Dense(
            iter::repeat(1.0).take(11)
                .chain(iter::repeat(0.0).take(9))
                .collect()
        );

        assert_eq!(p.n_features(), 20);

        assert_eq!(p.project(&[0.0]).unwrap(), output);
        assert_eq!(p.project(&[0.0, 1.0]).unwrap(), output);
        assert_eq!(p.project(&[-1.0, 1.0]).unwrap(), output);
    }
}
