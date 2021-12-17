use crate::ftzrs::*;
use crate::fuzzypoint::*;
use unidecode::unidecode;

use std::collections::{HashMap, HashSet};

use std::fmt::Debug;
use std::hash::{BuildHasherDefault, Hash, Hasher};
use std::marker::PhantomData;



pub trait HasFeatures {
    type Tok: Sized + Hash + Debug;
    fn collect_features_with<F: CanGram>(&self, ftzr: &F) -> Vec<Feature>;
}

pub trait HasLabel {
    type Label: Eq + Clone + Hash;
    fn label(&self) -> Self::Label;
}

impl HasFeatures for &str {
    type Tok = u8;
    fn collect_features_with<F: CanGram>(&self, ftzr: &F) -> Vec<Feature> {
        let mut feats: Vec<Feature> = Vec::with_capacity(10);
        {
            let mut updt = |f: Feature| feats.push(f);
            ftzr.run(unidecode(&self).as_bytes(), &mut updt); //TODO use chars
        }
        feats
    }
}

impl HasLabel for &str {
    type Label = Self;
    fn label(&self) -> Self {
        self
    }
}

impl HasFeatures for String {
    type Tok = u8;
    fn collect_features_with<F: CanGram>(&self, ftzr: &F) -> Vec<Feature> {
        self.as_str().collect_features_with(ftzr)
    }
}

impl HasLabel for String {
    type Label = Self;
    fn label(&self) -> Self {
        self.to_owned()
    }
}

impl HasFeatures for &String {
    type Tok = u8;
    fn collect_features_with<F: CanGram>(&self, ftzr: &F) -> Vec<Feature> {
        self.as_str().collect_features_with(ftzr)
    }
}

impl HasLabel for &String {
    type Label = String;
    fn label(&self) -> Self::Label {
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

impl<Tok: Sized + Hash + Debug> HasFeatures for &[Tok] {
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

impl<Tok: Sized + Hash + Debug, const N: usize> HasFeatures for [Tok; N] {
    type Tok = Tok;
    fn collect_features_with<F: CanGram>(&self, ftzr: &F) -> Vec<Feature> {
        let mut feats: Vec<Feature> = Vec::with_capacity(self.len());
        {
            let mut updt = |f: Feature| feats.push(f);
            ftzr.run(self, &mut updt);
        }
        feats
    }
}

impl<Tok: Sized + Hash + Debug, const N: usize> HasFeatures for &[Tok; N] {
    type Tok = Tok;
    fn collect_features_with<F: CanGram>(&self, ftzr: &F) -> Vec<Feature> {
        let mut feats: Vec<Feature> = Vec::with_capacity(self.len());
        {
            let mut updt = |f: Feature| feats.push(f);
            ftzr.run(*self, &mut updt);
        }
        feats
    }
}

impl<T: HasFeatures, Label> HasFeatures for Labeled<T, Label> {
    type Tok = <T as HasFeatures>::Tok;
    fn collect_features_with<F: CanGram>(&self, ftzr: &F) -> Vec<Feature> {
        self.point.collect_features_with(ftzr)
    }
}

impl<Point, Label: Eq + Hash + Clone> HasLabel for Labeled<Point, Label> {
    type Label = Label;
    fn label(&self) -> Self::Label {
        self.label.clone()
    }
}
