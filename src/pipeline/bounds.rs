use super::Transformer;
use num_traits::{clamp, NumOps};


pub struct LimitsTransformer<T> {
    lower: T,
    upper: T,
}

impl<T> LimitsTransformer<T> {
    fn new(lower: T, upper: T) -> LimitsTransformer<T> {
        LimitsTransformer {
            lower: lower,
            upper: upper,
        }
    }
}

impl<I: PartialOrd + Copy> Transformer<I, I> for LimitsTransformer<I> {
    fn transform(&self, input: I) -> Option<I> {
        Some(clamp(input, self.lower, self.upper))
    }
}


pub struct WrappingLimitsTransformer<T> {
    lower: T,
    upper: T,
}

impl<T> WrappingLimitsTransformer<T> {
    fn new(lower: T, upper: T) -> WrappingLimitsTransformer<T> {
        WrappingLimitsTransformer {
            lower: lower,
            upper: upper,
        }
    }

    #[inline]
    fn wrap_max(x: T, max: T) -> T where T: NumOps + Copy {
        (max + x % max) % max
    }
}

impl<I: NumOps + Copy> Transformer<I, I> for WrappingLimitsTransformer<I> {
    fn transform(&self, input: I) -> Option<I> {
        let offset = input - self.lower;
        let spread = self.upper - self.lower;

        Some(self.lower + WrappingLimitsTransformer::wrap_max(offset, spread))
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use std::f64::consts::PI;

    #[test]
    fn test_limits() {
        let wrapper = LimitsTransformer::new(0.0, 2.0*PI);

        assert_relative_eq!(wrapper.transform(0.0).unwrap(), 0.0);
        assert_relative_eq!(wrapper.transform(PI/3.0).unwrap(), PI/3.0);
        assert_relative_eq!(wrapper.transform(PI).unwrap(), PI);
        assert_relative_eq!(wrapper.transform(2.0*PI/3.0).unwrap(), 2.0*PI/3.0);
        assert_relative_eq!(wrapper.transform(2.0*PI).unwrap(), 2.0*PI);

        assert_relative_eq!(wrapper.transform(-4.0*PI).unwrap(), 0.0);
        assert_relative_eq!(wrapper.transform(4.0*PI).unwrap(), 2.0*PI);

        assert_relative_eq!(wrapper.transform(-0.12).unwrap(), 0.0);
        assert_relative_eq!(wrapper.transform(2.0*PI + 0.34).unwrap(), 2.0*PI);
    }

    #[test]
    fn test_wrapping_limits() {
        let wrapper = WrappingLimitsTransformer::new(0.0, 2.0*PI);

        assert_relative_eq!(wrapper.transform(0.0).unwrap(), 0.0);
        assert_relative_eq!(wrapper.transform(PI/3.0).unwrap(), PI/3.0);
        assert_relative_eq!(wrapper.transform(PI).unwrap(), PI);
        assert_relative_eq!(wrapper.transform(2.0*PI/3.0).unwrap(), 2.0*PI/3.0);
        assert_relative_eq!(wrapper.transform(2.0*PI).unwrap(), 0.0);

        assert_relative_eq!(wrapper.transform(-4.0*PI).unwrap(), 0.0);
        assert_relative_eq!(wrapper.transform(4.0*PI).unwrap(), 0.0);

        assert_relative_eq!(wrapper.transform(-0.12).unwrap(), 2.0*PI - 0.12);
        assert_relative_eq!(wrapper.transform(2.0*PI + 0.34).unwrap(), 0.34);
    }
}
