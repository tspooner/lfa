use crate::{
    core::*,
    eval::*,
    geometry::{Matrix, MatrixView, MatrixViewMut, Space},
    transforms::{Transform, Identity},
};

macro_rules! impl_builder {
    ($ftype:ty => $fname:ident) => {
        impl<P: Space, T> TransformedLFA<P, $ftype, T> {
            pub fn $fname(projector: P, transform: T) -> Self {
                let evaluator = <$ftype>::zeros(projector.dim());

                Self::new(projector, evaluator, transform)
            }
        }
    };
}

impl_builder!(ScalarFunction => scalar);
impl_builder!(PairFunction => pair);
impl_builder!(TripleFunction => triple);

/// Transformed linear function approximator.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct TransformedLFA<P, E, T = Identity> {
    pub projector: P,
    pub evaluator: E,
    pub transform: T,
}

impl<P, E, T> TransformedLFA<P, E, T> {
    pub fn new(projector: P, evaluator: E, transform: T) -> Self {
        TransformedLFA { projector, evaluator, transform, }
    }
}

impl<P: Space, T> TransformedLFA<P, VectorFunction, T> {
    pub fn vector(projector: P, n_outputs: usize, transform: T) -> Self {
        let evaluator = VectorFunction::zeros(projector.dim(), n_outputs);

        Self::new(projector, evaluator, transform)
    }
}

impl<P, E, T> TransformedLFA<P, E, T> {
    fn chain_update<V: Gradient>(&self, v: V, update: V) -> V where T: Transform<V> {
        self.transform.grad(v).chain(update)
    }
}

impl<P, E: Parameterised, T> Parameterised for TransformedLFA<P, E, T> {
    fn weights(&self) -> Matrix<f64> { self.evaluator.weights() }

    fn weights_view(&self) -> MatrixView<f64> { self.evaluator.weights_view() }

    fn weights_view_mut(&mut self) -> MatrixViewMut<f64> { self.evaluator.weights_view_mut() }
}

impl<P, E, T> Approximator for TransformedLFA<P, E, T>
where
    E: Approximator,
    T: Transform<E::Output>,
    E::Output: Gradient,
{
    type Output = E::Output;

    fn n_outputs(&self) -> usize {
        self.evaluator.n_outputs()
    }

    fn evaluate(&self, features: &Features) -> EvaluationResult<Self::Output> {
        self.evaluator.evaluate(features)
    }

    fn update(&mut self, features: &Features, update: Self::Output) -> UpdateResult<()> {
        match self.evaluator.evaluate(features).ok() {
            Some(v) => self.evaluator.update(features, self.chain_update(v, update)),
            None => Err(UpdateError::Failed),
        }
    }
}

impl<I: ?Sized, P: Projector<I>, E, T> Embedded<I> for TransformedLFA<P, E, T> {
    fn n_features(&self) -> usize {
        self.projector.dim()
    }

    fn to_features(&self, input: &I) -> Features {
        self.projector.project(input)
    }
}
