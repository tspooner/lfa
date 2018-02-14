#[derive(Debug)]
pub enum EvaluationError {

}

pub type EvaluationResult<T> = Result<T, EvaluationError>;

#[derive(Debug)]
pub enum UpdateError {

}

pub type UpdateResult<T> = Result<T, UpdateError>;
