use super::*;

#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
pub struct ISGD(pub f64);

impl ISGD {
    fn compute_lr(&self, features: &Features) -> f64 {
        let norm = features.fold(0.0, |acc, x| acc + x * x);

        self.0 / (1.0 + self.0 * norm)
    }
}

impl Optimiser<Features> for ISGD {
    fn step(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
    ) -> Result<()>
    {
        Ok(features.scaled_addto(self.compute_lr(features), weights))
    }

    fn step_scaled(
        &mut self,
        weights: &mut ArrayViewMut1<f64>,
        features: &Features,
        scale_factor: f64,
    ) -> Result<()>
    {
        Ok(features.scaled_addto(self.compute_lr(features) * scale_factor, weights))
    }
}
