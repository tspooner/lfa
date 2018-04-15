use geometry::{Vector, Space, Span};
use projectors::{IndexT, IndexSet, Feature, CandidateFeature};
use {Projection, Projector, AdaptiveProjector};

use std::collections::HashMap;
use rand::{Rng, ThreadRng, seq::sample_indices};
use itertools::Itertools;


pub struct IFDD<P: Projector<[f64]>> {
    pub base: P,
    pub features: Vec<Feature>,

    candidates: HashMap<IndexSet, CandidateFeature>,
    discovery_threshold: f64,
}

impl<P: Projector<[f64]>> IFDD<P> {
    pub fn new(base_projector: P, discovery_threshold: f64) -> Self {
        let initial_dim: usize = base_projector.dim();
        let mut base_features: Vec<Feature> = (0..initial_dim).map(|i| Feature {
            index: i,
            parent_indices: {
                let mut index_set = IndexSet::new();
                index_set.insert(i);
                index_set
            },
        }).collect();

        base_features.reserve(initial_dim);

        IFDD {
            base: base_projector,

            features: base_features,
            candidates: HashMap::new(),

            discovery_threshold: discovery_threshold,
        }
    }

    #[inline]
    fn inspect_candidate(&mut self, g: usize, h: usize, error: f64) -> Option<CandidateFeature> {
        let key = self.features[g].union(&self.features[h]);
        let rel = {
            let c = self.candidates
                .entry(key.clone()).or_insert_with(|| CandidateFeature::new(key.clone()));

            c.relevance += error.abs();

            c.relevance
        };

        if rel >= self.discovery_threshold {
            Some(self.candidates.remove(&key).unwrap())
        } else {
            None
        }
    }

    #[inline]
    fn discover_dense(&mut self, phi: Vector<f64>, error: f64) -> Vec<CandidateFeature> {
        (0..phi.len())
            .filter(|&i| phi[i].abs() < 1e-7)
            .combinations(2)
            .filter_map(|indices| self.inspect_candidate(indices[0], indices[1], error))
            .collect()
    }

    #[inline]
    fn discover_sparse(&mut self, active_indices: IndexSet, error: f64) -> Vec<CandidateFeature> {
        active_indices
            .iter()
            .tuple_combinations()
            .filter_map(|(&g, &h)| self.inspect_candidate(g, h, error))
            .collect()
    }
}

impl<P: Projector<[f64]>> Space for IFDD<P> {
    type Value = Projection;

    fn sample(&self, mut rng: &mut ThreadRng) -> Projection {
        let d = self.dim();
        let n = rng.gen_range(1, d);

        sample_indices(&mut rng, d, n).into()
    }

    fn dim(&self) -> usize { self.features.len() }

    fn span(&self) -> Span { unimplemented!() }
}

impl<P: Projector<[f64]>> Projector<[f64]> for IFDD<P> {
    fn project(&self, input: &[f64]) -> Projection {
        let mut p = self.project_base(input);
        let np: Vec<usize> = (self.base.dim()..self.dim()).filter_map(|i| {

            let f = &self.features[i];

            if f.parent_indices.iter().all(|i| p[*i].abs() < 1e-7) {
                Some(i)
            } else {
                None
            }

        }).collect();

        for i in np.iter() {
            for j in self.features[*i].parent_indices.iter() {
                p.remove(*j)
            }
        }

        p + np.into()
    }
}

impl<P: Projector<[f64]>> AdaptiveProjector<[f64]> for IFDD<P> {
    fn discover(&mut self, input: &[f64], error: f64) -> Option<HashMap<IndexT, IndexSet>> {
        use Projection::*;

        let new_features = match self.base.project(input) {
            Sparse(active_indices) => self.discover_sparse(active_indices, error),
            Dense(activations) => self.discover_dense(activations, error),
        };

        self.features.reserve_exact(new_features.len());

        new_features.into_iter().fold(None, |mut acc, f| {
            match self.add_feature(f) {
                Some(nf) => {
                    acc.get_or_insert_with(HashMap::new).insert(nf.0, nf.1);
                },
                None => (),
            };

            acc
        })
    }

    fn add_feature(&mut self, candidate: CandidateFeature) -> Option<(usize, IndexSet)> {
        let idx = self.features.len();
        let feature = candidate.into_feature(idx);
        let mapping = (idx, feature.parent_indices.clone());

        self.features.push(feature);

        Some(mapping)
    }

    fn project_base(&self, input: &[f64]) -> Projection {
        self.base.project(input)
    }
}


#[cfg(test)]
mod tests {
    extern crate seahash;

    use projectors::{fixed::TileCoding, adaptive::IFDD};
    use super::*;
    use std::hash::BuildHasherDefault;

    type SHBuilder = BuildHasherDefault<seahash::SeaHasher>;

    #[derive(Clone)]
    struct BaseProjector;

    impl Space for BaseProjector {
        type Value = Projection;

        fn sample(&self, _: &mut ThreadRng) -> Projection { unimplemented!() }

        fn dim(&self) -> usize { 5 }

        fn span(&self) -> Span { unimplemented!() }
    }

    impl Projector<[f64]> for BaseProjector {
        fn project(&self, input: &[f64]) -> Projection {
            input.iter().map(|v| v.round().min(4.0).max(0.0) as usize).collect()
        }
    }

    #[test]
    fn test_discover() {
        let mut f = IFDD::new(BaseProjector, 10.0);

        assert_eq!(f.discover(&vec![0.0, 4.0], 10.0), Some({
            let mut hm = HashMap::new();
            hm.insert(5, [0, 4].iter().cloned().collect());
            hm
        }));
        assert_eq!(f.features[5], Feature {
            index: 5,
            parent_indices: [0, 4].iter().cloned().collect(),
        });

        assert_eq!(f.discover(&vec![0.0, 3.0], 5.0), None);
        assert_eq!(f.features.len(), 6);

        assert_eq!(f.discover(&vec![0.0, 3.0], 5.0), Some({
            let mut hm = HashMap::new();
            hm.insert(6, [0, 3].iter().cloned().collect());
            hm
        }));
        assert_eq!(f.features[6], Feature {
            index: 6,
            parent_indices: [0, 3].iter().cloned().collect(),
        });
    }

    #[test]
    fn test_project_base() {
        let b = TileCoding::new(SHBuilder::default(), 8, 100);
        let f = IFDD::new(b.clone(), 100.0);

        assert_eq!(b.project(&vec![0.0, 1.0]), f.project_base(&vec![0.0, 1.0]));

        let b = BaseProjector;
        let f = IFDD::new(b.clone(), 100.0);

        assert_eq!(b.project(&vec![0.0, 1.0]), f.project_base(&vec![0.0, 1.0]));
    }
}
