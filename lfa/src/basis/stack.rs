use crate::{IndexT, ActivationT, Features, Result, Error, basis::Projector};

/// Stack the output of two `Projector` instances.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct Stacker<P1, P2> {
    p1: P1,
    p2: P2,
}

impl<P1, P2> Stacker<P1, P2> {
    pub fn new(p1: P1, p2: P2) -> Self { Stacker { p1, p2 } }
}

impl<P1, P2> Projector for Stacker<P1, P2>
where
    P1: Projector,
    P2: Projector,
{
    fn n_features(&self) -> usize {
        self.p1.n_features() + self.p2.n_features()
    }

    fn project_ith(&self, input: &[f64], index: IndexT) -> Result<Option<ActivationT>> {
        let n1 = self.p1.n_features();
        let n2 = self.p2.n_features();
        let n12 = n1 + n2;

        if index < n1 {
            self.p1.project_ith(input, index)
        } else if index < n12 {
            self.p2.project_ith(input, index - n1)
        } else {
            Err(Error::index_error(index, n12))
        }
    }

    fn project(&self, input: &[f64]) -> Result<Features> {
        self.p1.project(input).and_then(|f1| {
            self.p2.project(input).map(|f2| f1.stack(f2))
        })
    }
}

#[cfg(test)]
mod tests {
    use crate::basis::{Projector, Constants, Indices};
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
