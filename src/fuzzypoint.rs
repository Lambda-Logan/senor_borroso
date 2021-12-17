use crate::dualiter::*;
use crate::ftzrs::{CanGram, Feature};
use crate::hasfeatures::HasFeatures;
use fxhash::FxHasher64;
use std::cell::Cell;
use std::cmp;
use std::hash::{Hash, Hasher};
use std::rc::Rc;

pub trait FromFeatures {
    fn made_from<Origin: HasFeatures, Ftzr: CanGram>(origin: &Origin, ftzr: &Ftzr) -> Self;
}

pub trait FuzzyPoint {
    fn get_sorted_features<'a>(&'a self) -> &'a [Feature];

    fn get_features_mut<'a>(&'a mut self) -> &'a mut Vec<Feature>;

    fn len(&self) -> usize {
        self.get_sorted_features().len()
    }
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct SimplePoint {
    feats: Vec<Feature>,
}

impl SimplePoint {
    pub fn new(feats: Vec<Feature>) -> Self {
        let mut _feats = feats;
        _feats.sort();
        SimplePoint { feats: _feats }
    }
}

impl FromFeatures for SimplePoint {
    fn made_from<Origin: HasFeatures, Ftzr: CanGram>(origin: &Origin, ftzr: &Ftzr) -> Self {
        let mut feats = origin.collect_features_with(ftzr);
        feats.sort();
        SimplePoint { feats: feats }
    }
}

impl FuzzyPoint for SimplePoint {
    fn get_sorted_features<'a>(&'a self) -> &'a [Feature] {
        &self.feats
    }

    fn get_features_mut<'a>(&'a mut self) -> &'a mut Vec<Feature> {
        &mut self.feats
    }
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq, Hash)]
pub struct Labeled<Point, Id = String> {
    pub label: Id,
    pub point: Point,
}

impl<Point, Label> Labeled<Point, Label> {
    pub fn new(label: Label, point: Point) -> Self {
        Labeled {
            label: label,
            point: point,
        }
    }

    pub fn from_tuple(pair: (Label, Point)) -> Self {
        Labeled::new(pair.0, pair.1)
    }
}

impl<Point> Labeled<Point, ()> {
    pub fn anon(point: Point) -> Self {
        Labeled {
            label: (),
            point: point,
        }
    }
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
pub struct SimHash {
    pub(crate) hash: u64,
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

impl FromFeatures for SimHash {
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
}

impl FuzzyPoint for SimHash {
    fn get_sorted_features<'a>(&'a self) -> &'a [Feature] {
        &self.feats
    }

    fn get_features_mut<'a>(&'a mut self) -> &'a mut Vec<Feature> {
        &mut self.feats
    }
}
