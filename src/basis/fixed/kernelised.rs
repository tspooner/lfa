use crate::basis::{Composable, Projection, Projector, kernels::{self, Kernel}};
use crate::geometry::{discrete::Partition, product::LinearSpace, Card, Space, Vector};
use crate::utils::cartesian_product;

/// Feature prototype used by the `KernelProjector` basis.
#[derive(Clone, Copy, Serialize, Deserialize, Debug)]
pub struct Prototype<I, K> {
    pub centroid: I,
    pub kernel: K,
}

impl<I, K: Kernel<I>> Prototype<I, K> {
    pub fn kernel(&self, x: &I) -> f64 { self.kernel.kernel(x, &self.centroid) }
}

/// Kernel machine basis projector.
#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct KernelProjector<I, K> {
    pub prototypes: Vec<Prototype<I, K>>,
}

impl<I, K: Kernel<I>> KernelProjector<I, K> {
    pub fn new(prototypes: Vec<Prototype<I, K>>) -> Self { KernelProjector { prototypes } }

    pub fn homogeneous(centroids: impl IntoIterator<Item = impl Into<I>>, kernel: K) -> Self {
        KernelProjector::new(
            centroids
                .into_iter()
                .map(|c| Prototype {
                    centroid: c.into(),
                    kernel: kernel.clone(),
                })
                .collect(),
        )
    }
}

impl KernelProjector<Vector<f64>, kernels::ExpQuad> {
    pub fn exp_quad(partitioning: LinearSpace<Partition>) -> Self {
        let lengthscales = partitioning.iter().map(|d| d.partition_width()).collect();
        let kernel = kernels::ExpQuad::new(1.0, lengthscales);
        let centroids = cartesian_product(&partitioning.centres());

        KernelProjector::homogeneous(centroids, kernel)
    }
}

impl KernelProjector<Vector<f64>, kernels::Matern32> {
    pub fn matern_32(partitioning: LinearSpace<Partition>) -> Self {
        let lengthscales = partitioning.iter().map(|d| d.partition_width()).collect();
        let kernel = kernels::Matern32::new(1.0, lengthscales);
        let centroids = cartesian_product(&partitioning.centres());

        KernelProjector::homogeneous(centroids, kernel)
    }
}

impl KernelProjector<Vector<f64>, kernels::Matern52> {
    pub fn matern_52(partitioning: LinearSpace<Partition>) -> Self {
        let lengthscales = partitioning.iter().map(|d| d.partition_width()).collect();
        let kernel = kernels::Matern52::new(1.0, lengthscales);
        let centroids = cartesian_product(&partitioning.centres());

        KernelProjector::homogeneous(centroids, kernel)
    }
}

impl<I, K: Kernel<I>> Space for KernelProjector<I, K> {
    type Value = Projection;

    fn dim(&self) -> usize { self.prototypes.len() }

    fn card(&self) -> Card { Card::Infinite }
}

impl<I, K: Kernel<I>> Projector<I> for KernelProjector<I, K> {
    fn project(&self, input: &I) -> Projection {
        Projection::Dense(self.prototypes.iter().map(|p| p.kernel(input)).collect())
    }
}

impl<I, K: Kernel<I>> Composable for KernelProjector<I, K> {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geometry::Vector;
    use crate::kernels::ExpQuad;

    fn make_net(centroids: Vec<Vec<f64>>, ls: Vec<f64>) -> KernelProjector<Vector<f64>, ExpQuad> {
        let kernel = ExpQuad::new(1.0, Vector::from_vec(ls));

        KernelProjector::homogeneous(centroids, kernel)
    }

    fn make_net_1d(centroids: Vec<f64>, ls: f64) -> KernelProjector<Vector<f64>, ExpQuad> {
        make_net(centroids.into_iter().map(|c| vec![c]).collect(), vec![ls])
    }

    #[test]
    fn test_dimensionality() {
        assert_eq!(make_net_1d(vec![0.0], 0.25).dim(), 1);
        assert_eq!(make_net_1d(vec![0.0, 0.5, 1.0], 0.25).dim(), 3);
        assert_eq!(make_net_1d(vec![0.0; 10], 0.25).dim(), 10);
        assert_eq!(make_net_1d(vec![0.0; 100], 0.25).dim(), 100);
    }

    #[test]
    fn test_cardinality() {
        assert_eq!(make_net_1d(vec![0.0], 0.25).card(), Card::Infinite);
        assert_eq!(
            make_net_1d(vec![0.0, 0.5, 1.0], 0.25).card(),
            Card::Infinite
        );
        assert_eq!(make_net_1d(vec![0.0; 10], 0.25).card(), Card::Infinite);
        assert_eq!(make_net_1d(vec![0.0; 100], 0.25).card(), Card::Infinite);
    }

    #[test]
    fn test_projection_1d() {
        let net = make_net_1d(vec![0.0, 0.5, 1.0], 0.25);
        let p = net.project_expanded(&Vector::from_vec(vec![0.25]));

        assert!(p.all_close(
            &Vector::from_vec(vec![0.6065307, 0.6065307, 0.0111090]),
            1e-6
        ));
    }

    #[test]
    fn test_projection_2d() {
        let net = make_net(
            vec![vec![0.0, -10.0], vec![0.5, -8.0], vec![1.0, -6.0]],
            vec![0.25, 2.0],
        );
        let p = net.project_expanded(&Vector::from_vec(vec![0.67, -7.0]));

        assert!(p.all_close(
            &Vector::from_vec(vec![0.0089491, 0.7003325, 0.369280]),
            1e-6
        ));
    }
}
