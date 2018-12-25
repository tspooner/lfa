use crate::basis::{Projector, Projection};
use crate::geometry::{
    Space, Card,
    product::LinearSpace,
    discrete::Partition,
    continuous::Reals,
};
use crate::kernels::{self, Kernel};
use crate::utils::cartesian_product;


pub type RealLinearSpaceVec = <LinearSpace<Reals> as Space>::Value;
pub type RBFNetwork = KernelProjector<RealLinearSpaceVec, kernels::ExpQuad>;


#[derive(Clone, Serialize, Deserialize)]
pub struct Prototype<I, K: Kernel<I>> {
    pub kernel: K,
    pub centroid: I,
}

impl<I, K: Kernel<I>> Prototype<I, K> {
    pub fn kernel(&self, x: &I) -> f64 {
        self.kernel.kernel(x, &self.centroid)
    }
}

#[derive(Clone)]
pub struct KernelProjector<I, K: Kernel<I>> {
    pub prototypes: Vec<Prototype<I, K>>,
}

impl<I, K: Kernel<I>> KernelProjector<I, K> {
    pub fn new(prototypes: Vec<Prototype<I, K>>) -> Self {
        KernelProjector { prototypes }
    }

    pub fn from_centroids(centroids: impl IntoIterator<Item = impl Into<I>>, kernel: K) -> Self {
        KernelProjector::new(centroids
            .into_iter()
            .map(|c| Prototype {
                kernel: kernel.clone(),
                centroid: c.into(),
            })
            .collect()
        )
    }
}

impl<K: Kernel<RealLinearSpaceVec>> KernelProjector<RealLinearSpaceVec, K> {
    pub fn from_partitioning(partitioning: LinearSpace<Partition>, kernel: K) -> Self {
        let centroids = cartesian_product(&partitioning.centres());

        KernelProjector::from_centroids(centroids, kernel)
    }
}

impl KernelProjector<RealLinearSpaceVec, kernels::ExpQuad> {
    pub fn rbf_network(partitioning: LinearSpace<Partition>) -> Self {
        let lengthscales = partitioning.iter().map(|d| d.partition_width()).collect();
        let kernel = kernels::RBF::new(1.0, lengthscales);
        let centroids = cartesian_product(&partitioning.centres());

        KernelProjector::from_centroids(centroids, kernel)
    }
}

impl<I, K: Kernel<I>> Space for KernelProjector<I, K> {
    type Value = Projection;

    fn dim(&self) -> usize {
        self.prototypes.len()
    }

    fn card(&self) -> Card {
        Card::Infinite
    }
}

impl<I, K: Kernel<I>> Projector<I> for KernelProjector<I, K> {
    fn project(&self, input: &I) -> Projection {
        Projection::Dense(self.prototypes.iter().map(|p| p.kernel(input)).collect())
    }
}


#[cfg(test)]
mod tests {
    use crate::geometry::Vector;
    use super::*;

    fn make_net(centroids: Vec<Vec<f64>>, ls: Vec<f64>) -> RBFNetwork {
        let kernel = kernels::ExpQuad::new(1.0, Vector::from_vec(ls));

        KernelProjector::from_centroids(centroids, kernel)
    }

    fn make_net_1d(centroids: Vec<f64>, ls: f64) -> RBFNetwork {
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
        assert_eq!(make_net_1d(vec![0.0, 0.5, 1.0], 0.25).card(), Card::Infinite);
        assert_eq!(make_net_1d(vec![0.0; 10], 0.25).card(), Card::Infinite);
        assert_eq!(make_net_1d(vec![0.0; 100], 0.25).card(), Card::Infinite);
    }

    #[test]
    fn test_projection_1d() {
        let net = make_net_1d(vec![0.0, 0.5, 1.0], 0.25);
        let p = net.project_expanded(&Vector::from_vec(vec![0.25]));

        assert!(p.all_close(&Vector::from_vec(vec![0.6065307, 0.6065307, 0.0111090]), 1e-6));
    }

    #[test]
    fn test_projection_2d() {
        let net = make_net(
            vec![vec![0.0, -10.0], vec![0.5, -8.0], vec![1.0, -6.0]],
            vec![0.25, 2.0],
        );
        let p = net.project_expanded(&Vector::from_vec(vec![0.67, -7.0]));

        assert!(p.all_close(&Vector::from_vec(vec![0.0089491, 0.7003325, 0.369280]), 1e-6));
    }
}
