use crate::{
    basis::Projector,
    core::*,
    geometry::{Card, Space, Vector},
};
use itertools::Itertools;
use std::{
    cmp::Ordering,
    collections::HashMap,
    hash::{Hash, Hasher},
};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct Feature {
    pub index: usize,
    pub parent_indices: IndexSet,
}

impl Feature {
    pub fn union(&self, other: &Self) -> IndexSet {
        let g = &self.parent_indices;
        let h = &other.parent_indices;

        g.union(h).cloned().collect()
    }
}

impl Hash for Feature {
    fn hash<H: Hasher>(&self, state: &mut H) { self.index.hash(state); }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct CandidateFeature {
    pub relevance: f64,
    pub parent_indices: IndexSet,
}

impl CandidateFeature {
    pub fn new<T: Into<IndexSet>>(parent_indices: T) -> Self {
        CandidateFeature {
            relevance: 0.0,
            parent_indices: parent_indices.into(),
        }
    }

    pub fn from_vec(index_vec: Vec<usize>) -> Self {
        let mut parent_indices = IndexSet::new();

        for i in index_vec {
            parent_indices.insert(i);
        }

        CandidateFeature::new(parent_indices)
    }

    pub fn into_feature(self, index: usize) -> Feature {
        Feature {
            index: index,
            parent_indices: self.parent_indices,
        }
    }
}

impl Hash for CandidateFeature {
    fn hash<H: Hasher>(&self, state: &mut H) { self.parent_indices.hash(state); }
}

impl PartialOrd for CandidateFeature {
    fn partial_cmp(&self, other: &CandidateFeature) -> Option<Ordering> {
        self.relevance.partial_cmp(&other.relevance)
    }
}

#[derive(Clone, Serialize, Deserialize, Debug)]
pub struct IFDD<P> {
    pub base: P,
    pub features: Vec<Feature>,

    candidates: HashMap<IndexSet, CandidateFeature>,
    discovery_threshold: f64,
}

impl<P: Space> IFDD<P> {
    pub fn new(base_projector: P, discovery_threshold: f64) -> Self {
        let initial_dim: usize = base_projector.dim();
        let mut base_features: Vec<Feature> = (0..initial_dim)
            .map(|i| Feature {
                index: i,
                parent_indices: {
                    let mut index_set = IndexSet::new();
                    index_set.insert(i);
                    index_set
                },
            })
            .collect();

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
            let c = self
                .candidates
                .entry(key.clone())
                .or_insert_with(|| CandidateFeature::new(key.clone()));

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

    fn add_feature(&mut self, candidate: CandidateFeature) -> Option<(usize, IndexSet)> {
        let idx = self.features.len();
        let feature = candidate.into_feature(idx);
        let mapping = (idx, feature.parent_indices.clone());

        self.features.push(feature);

        Some(mapping)
    }
}

impl<P> IFDD<P> {
    fn discover<I>(&mut self, input: &I, error: f64) -> Option<HashMap<IndexT, IndexSet>>
    where
        I: ?Sized,
        P: Projector<I>
    {
        let new_features = match self.base.project(input) {
            Features::Sparse(active_indices) => self.discover_sparse(active_indices, error),
            Features::Dense(activations) => self.discover_dense(activations, error),
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
}

impl<P: Space> Space for IFDD<P> {
    type Value = Features;

    fn dim(&self) -> usize { self.features.len() }

    fn card(&self) -> Card { unimplemented!() }
}

impl<I: ?Sized, P: Projector<I>> Projector<I> for IFDD<P> {
    fn project(&self, input: &I) -> Features {
        let mut p = self.base.project(input);
        let np: Vec<usize> = (self.base.dim()..self.dim())
            .filter_map(|i| {
                let f = &self.features[i];

                if f.parent_indices.iter().all(|i| p[*i].abs() < 1e-7) {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();

        for i in np.iter() {
            for j in self.features[*i].parent_indices.iter() {
                p.remove(*j)
            }
        }

        p + np.into()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basis::ifdd::IFDD;

    #[derive(Clone)]
    struct BaseProjector;

    impl Space for BaseProjector {
        type Value = Features;

        fn dim(&self) -> usize { 5 }

        fn card(&self) -> Card { unimplemented!() }
    }

    impl Projector<[f64]> for BaseProjector {
        fn project(&self, input: &[f64]) -> Features {
            input
                .iter()
                .map(|v| v.round().min(4.0).max(0.0) as usize)
                .collect()
        }
    }

    #[test]
    fn test_discover() {
        let mut f = IFDD::new(BaseProjector, 10.0);

        assert_eq!(
            f.discover(&vec![0.0, 4.0], 10.0),
            Some({
                let mut hm = HashMap::new();
                hm.insert(5, [0, 4].iter().cloned().collect());
                hm
            })
        );
        assert_eq!(
            f.features[5],
            Feature {
                index: 5,
                parent_indices: [0, 4].iter().cloned().collect(),
            }
        );

        assert_eq!(f.discover(&vec![0.0, 3.0], 5.0), None);
        assert_eq!(f.features.len(), 6);

        assert_eq!(
            f.discover(&vec![0.0, 3.0], 5.0),
            Some({
                let mut hm = HashMap::new();
                hm.insert(6, [0, 3].iter().cloned().collect());
                hm
            })
        );
        assert_eq!(
            f.features[6],
            Feature {
                index: 6,
                parent_indices: [0, 3].iter().cloned().collect(),
            }
        );
    }
}
