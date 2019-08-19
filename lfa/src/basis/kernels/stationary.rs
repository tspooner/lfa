macro_rules! stationary_kernel {
    ($(#[$attr:meta])* => $name:ident, $self:ident, $r:ident, $code:block) => {
        $(#[$attr])*
        #[derive(Clone)]
        #[cfg_attr(feature = "serialize", derive(Serialize, Deserialize))]
        pub struct $name {
            pub variance: f64,
            pub lengthscales: Vec<f64>,
        }

        impl $name {
            pub fn new(variance: f64, lengthscales: Vec<f64>) -> $name {
                $name { variance, lengthscales }
            }

            pub fn non_ard(variance: f64, lengthscale: f64) -> $name {
                $name::new(variance, vec![lengthscale])
            }

            fn kernel_stationary(&$self, $r: f64) -> f64 $code
        }

        impl Default for $name {
            fn default() -> $name {
                $name::non_ard(1.0, 1.0)
            }
        }

        impl super::Kernel<f64> for $name {
            fn kernel(&$self, x: &f64, y: &f64) -> f64 {
                $self.kernel_stationary((x - y).abs() / $self.lengthscales[0])
            }
        }

        impl super::Kernel<[f64]> for $name {
            fn kernel(&$self, x: &[f64], y: &[f64]) -> f64 {
                $self.kernel_stationary(x
                    .into_iter()
                    .zip(y.into_iter())
                    .map(|(x, y)| x - y)
                    .zip($self.lengthscales.iter())
                    .map(|(d, l)| d / l)
                    .fold(0.0f64, |acc, z| acc + z * z)
                    .sqrt())
            }
        }

        impl super::Kernel<Vec<f64>> for $name {
            fn kernel(&$self, x: &Vec<f64>, y: &Vec<f64>) -> f64 {
                super::Kernel::<[f64]>::kernel($self, x, y)
            }
        }
    };
}

stationary_kernel!(
    /// Exponential kernel.
    => Exponential, self, r, {
        self.variance * (-r).exp()
    }
);

stationary_kernel!(
    /// Matern 3/2 kernel.
    => Matern32, self, r, {
        let sqrt_3 = 3.0f64.sqrt();

        self.variance * (1.0 + sqrt_3 * r) * (-sqrt_3 * r).exp()
    }
);

stationary_kernel!(
    /// Matern 5/2 kernel.
    => Matern52, self, r, {
        let sqrt_5 = 3.0f64.sqrt();

        self.variance * (1.0 + sqrt_5 * r + 5.0 / 3.0 * r * r) * (-sqrt_5 * r).exp()
    }
);

stationary_kernel!(
    /// Exponentiated quadratic kernel.
    => ExpQuad, self, r, {
        self.variance * (-0.5 * r * r).exp()
    }
);

stationary_kernel!(
    /// Cosine kernel.
    => Cosine, self, r, {
        self.variance * r.cos()
    }
);

pub type RBF = ExpQuad;
pub type Guassian = ExpQuad;
