use crate::ftzrs::*;
use crate::fuzzypoint::*;
use unidecode::unidecode;

use std::fmt::Debug;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;

pub trait HasFeatures {
    type Tok: Sized + Hash + Debug;
    fn collect_features_with<F: CanGram>(&self, ftzr: &F) -> Vec<Feature>;
}

pub trait HasName {
    type Id: Eq + Clone + Hash;
    fn name(&self) -> Self::Id;
}

impl HasFeatures for &str {
    type Tok = u8;
    fn collect_features_with<F: CanGram>(&self, ftzr: &F) -> Vec<Feature> {
        let mut feats: Vec<Feature> = Vec::with_capacity(10);
        {
            let mut updt = |f: Feature| feats.push(f);
            //let chars: Vec<_> = Iterator::collect(self.chars());
            //ftzr.run(&chars, &mut updt);
            ftzr.run(unidecode(&self).as_bytes(), &mut updt); //TODO use chars
        }
        feats
    }
}

impl HasName for &str {
    type Id = Self;
    fn name(&self) -> Self {
        self
    }
}

impl HasFeatures for String {
    type Tok = u8;
    fn collect_features_with<F: CanGram>(&self, ftzr: &F) -> Vec<Feature> {
        self.as_str().collect_features_with(ftzr)
    }
}

impl HasName for String {
    type Id = Self;
    fn name(&self) -> Self {
        self.to_owned()
    }
}

impl HasFeatures for &String {
    type Tok = u8;
    fn collect_features_with<F: CanGram>(&self, ftzr: &F) -> Vec<Feature> {
        self.as_str().collect_features_with(ftzr)
    }
}

impl HasName for &String {
    type Id = String;
    fn name(&self) -> Self::Id {
        self.to_string()
    }
}

impl<Tok: Sized + Hash + Debug> HasFeatures for [Tok] {
    type Tok = Tok;
    fn collect_features_with<F: CanGram>(&self, ftzr: &F) -> Vec<Feature> {
        let mut feats: Vec<Feature> = Vec::with_capacity(self.len());
        {
            let mut updt = |f: Feature| feats.push(f);
            ftzr.run(&self, &mut updt);
        }
        feats
    }
}
