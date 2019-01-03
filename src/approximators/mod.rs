//! Module for efficient dot product abstractions used by `LFA`.
use crate::core::*;
use crate::geometry::Matrix;
use std::{collections::HashMap, mem::replace};

/// An interface for function approximators.
pub trait Approximator<I: ?Sized> {
    type Value;

    /// Evaluate the function and return its value.
    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Value>;

    /// Update the approximator's estimate for the given input.
    fn update(&mut self, input: &I, update: Self::Value) -> UpdateResult<()>;

    #[allow(unused_variables)]
    /// Adapt the approximator in light of newly discovered features.
    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        unimplemented!()
    }
}

impl<I: ?Sized, T: Approximator<I>> Approximator<I> for Box<T> {
    type Value = T::Value;

    fn evaluate(&self, input: &I) -> EvaluationResult<Self::Value> { (**self).evaluate(input) }

    fn update(&mut self, input: &I, update: Self::Value) -> UpdateResult<()> {
        (**self).update(input, update)
    }

    fn adapt(&mut self, new_features: &HashMap<IndexT, IndexSet>) -> AdaptResult<usize> {
        (**self).adapt(new_features)
    }
}

fn append_matrix_rows(weights: &mut Matrix<f64>, new_rows: Vec<Vec<f64>>) {
    let n_cols = weights.cols();
    let n_rows = weights.rows();
    let n_rows_new = new_rows.len();

    // Weight matrix stored in row-major format.
    let mut new_weights = unsafe { replace(weights, Matrix::uninitialized((0, 0))).into_raw_vec() };

    new_weights.reserve_exact(n_rows_new);

    for row in new_rows {
        new_weights.extend(row);
    }

    *weights = Matrix::from_shape_vec((n_rows + n_rows_new, n_cols), new_weights).unwrap();
}

fn adapt_matrix(
    weights: &mut Matrix<f64>,
    new_features: &HashMap<IndexT, IndexSet>,
) -> AdaptResult<usize>
{
    let n_nfs = new_features.len();
    let n_outputs = weights.cols();
    let max_index = weights.len() + n_nfs - 1;

    let new_weights: Result<Vec<Vec<f64>>, _> = new_features
        .into_iter()
        .map(|(&i, idx)| {
            if i > max_index {
                Err(AdaptError::Failed)
            } else {
                Ok((0..n_outputs)
                    .map(|c| {
                        let c = weights.column(c);

                        idx.iter().fold(0.0, |acc, r| acc + c[*r])
                    })
                    .collect())
            }
        })
        .collect();

    match new_weights {
        Ok(new_weights) => {
            append_matrix_rows(weights, new_weights);

            Ok(n_nfs)
        },
        Err(err) => Err(err),
    }
}

import_all!(scalar);
import_all!(pair);
import_all!(triple);
import_all!(vector);
