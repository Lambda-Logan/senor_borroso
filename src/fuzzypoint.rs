use crate::dualiter::*;
use crate::ftzrs::{CanGram, Feature};
use crate::pointfactories::HasFeatures;
use fxhash::FxHasher64;
use space::MetricPoint;
use std::cell::Cell;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

pub trait FuzzyPoint {
    fn made_from<Origin: HasFeatures, Ftzr: CanGram>(origin: &Origin, ftzr: &Ftzr) -> Self;

    fn get_sorted_features<'a>(&'a self) -> &'a [Feature];

    fn get_features_mut<'a>(&'a mut self) -> &'a mut Vec<Feature>;
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct SimplePoint {
    pub feats: Vec<Feature>,
}

impl FuzzyPoint for SimplePoint {
    fn made_from<Origin: HasFeatures, Ftzr: CanGram>(origin: &Origin, ftzr: &Ftzr) -> Self {
        let mut feats = origin.collect_features_with(ftzr);
        feats.sort();
        SimplePoint { feats: feats }
    }

    fn get_sorted_features<'a>(&'a self) -> &'a [Feature] {
        &self.feats
    }

    fn get_features_mut<'a>(&'a mut self) -> &'a mut Vec<Feature> {
        &mut self.feats
    }
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct Labeled<Point, Id = String> {
    pub label: Id,
    pub point: Point,
}

pub trait Metric<P> {
    fn sim(&self, p1: &P, p2: &P) -> f64;
    fn dist(&self, p1: &P, p2: &P) -> f64;
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct Jaccard;

impl<P: FuzzyPoint> Metric<P> for Jaccard {
    fn dist(&self, p1: &P, p2: &P) -> f64 {
        let sim = self.sim(p1, p2);
        if sim > 0.0 {
            1.0 / sim
        } else {
            f64::MAX
        }
    }

    #[inline(always)]
    fn sim(&self, p1: &P, p2: &P) -> f64 {
        //return 1.0 / self.ham_dist(other) as f64;
        let mut n = 0;
        let mut d = 0;
        for x in DualIter::new(
            p1.get_sorted_features().iter(),
            p2.get_sorted_features().iter(),
        ) {
            match x {
                AndOrOr::Both(_, _) => {
                    d += 1;
                    n += 1;
                }
                _ => {
                    d += 1;
                }
            }
        }
        if n == 0 || d == 0 {
            return 0.0;
        }
        n as f64 / d as f64
    }
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct Counted<M> {
    pub metric: M,
    pub tally: Rc<Cell<usize>>,
}

impl<M> Counted<M> {
    pub fn new(m: M) -> Self {
        Counted {
            metric: m,
            tally: Rc::new(Cell::new(0)),
        }
    }
}

impl<P, M: Metric<P>> Metric<P> for Counted<M> {
    fn sim(&self, p1: &P, p2: &P) -> f64 {
        let t = self.tally.get();
        self.tally.set(t + 1);
        self.metric.sim(p1, p2)
    }
    fn dist(&self, p1: &P, p2: &P) -> f64 {
        let t = self.tally.get();
        self.tally.set(t + 1);
        self.metric.dist(p1, p2)
    }
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct SimHash {
    hash: u64,
    feats: Vec<Feature>,
}

fn hash_feature<T: Hash>(t: &T) -> u64 {
    let mut hasher = FxHasher64::default();
    t.hash(&mut hasher);
    hasher.finish()
}

/// Calculate `u64` simhash from stream of `&str` words
pub fn simhash_stream<W, T: Hash>(words: W) -> u64
where
    W: Iterator<Item = T>,
{
    let mut v = [0i32; 64];
    let mut simhash: u64 = 0;

    for feature in words {
        let feature_hash: u64 = hash_feature(&feature);

        for i in 0..64 {
            let bit = (feature_hash >> i) & 1;
            if bit == 1 {
                v[i] = v[i].saturating_add(1);
            } else {
                v[i] = v[i].saturating_sub(1);
            }
        }
    }

    for q in 0..64 {
        if v[q] > 0 {
            simhash |= 1 << q;
        }
    }
    simhash
}

pub fn simhash_stream_u128<W, T: Hash>(words: W) -> u128
where
    W: Iterator<Item = T>,
{
    let mut v = [0i32; 128];
    let mut simhash: u128 = 0;

    for feature in words {
        let feature_hash: u64 = hash_feature(&feature);

        for i in 0..128 {
            let bit = (feature_hash >> i) & 1;
            if bit == 1 {
                v[i] = v[i].saturating_add(1);
            } else {
                v[i] = v[i].saturating_sub(1);
            }
        }
    }

    for q in 0..128 {
        if v[q] > 0 {
            simhash |= 1 << q;
        }
    }
    simhash
}

impl FuzzyPoint for SimHash {
    fn made_from<Origin: HasFeatures, Ftzr: CanGram>(origin: &Origin, ftzr: &Ftzr) -> Self {
        let mut feats: Vec<Feature> = origin.collect_features_with(ftzr);
        //feats.sort();
        let sh = simhash_stream(feats.iter());
        feats.sort();
        SimHash {
            hash: sh,
            feats: feats,
        }
    }

    fn get_sorted_features<'a>(&'a self) -> &'a [Feature] {
        &self.feats
    }

    fn get_features_mut<'a>(&'a mut self) -> &'a mut Vec<Feature> {
        &mut self.feats
    }
}
#[derive(Debug, Copy, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct Hamming;

impl Metric<SimHash> for Hamming {
    fn sim(&self, p1: &SimHash, p2: &SimHash) -> f64 {
        (1.0 / (self.dist(p1, p2) + 1.0)).sqrt()
    }
    fn dist(&self, p1: &SimHash, p2: &SimHash) -> f64 {
        (p1.hash ^ p2.hash).count_ones() as f64
    }
}

pub struct HnswPoint<M: Metric<P>, P: FuzzyPoint> {
    metric: M,
    point: P,
}

impl<P: FuzzyPoint, M: Metric<P>> HnswPoint<M, P> {
    pub fn new(m: M, p: P) -> Self {
        HnswPoint {
            metric: m,
            point: p,
        }
    }
}

impl<P: FuzzyPoint, M: Metric<P>> MetricPoint for HnswPoint<M, P> {
    fn distance(&self, rhs: &Self) -> u64 {
        self.metric.dist(&self.point, &rhs.point) as u64
    }
}
