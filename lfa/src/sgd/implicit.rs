use super::*;

pub struct ISGD(pub f64);

impl Optimiser<Features> for ISGD {
    fn step(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        error: f64
    ) -> UpdateResult<()> {
        let norm = features.fold(0.0, |acc, x| acc + x*x);
        let lr = self.0 / (1.0 + self.0 * norm);

        Ok(features.scaled_addto(lr * error, weights))
    }
}
