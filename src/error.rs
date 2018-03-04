#[derive(Debug)]
pub enum EvaluationError {
    Failed,
}

pub type EvaluationResult<T> = Result<T, EvaluationError>;


#[derive(Debug)]
pub enum UpdateError {
    Failed,
}

pub type UpdateResult<T> = Result<T, UpdateError>;
