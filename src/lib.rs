#![allow(warnings)]
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::str;
use unidecode::unidecode;

mod testing;
use testing::{FuzzyIndexTest, HnswTester, StatsPad, Testable, TrainingAtom};

mod feat;
//use feat::{
//    book_ends, n_gram, skipgram, AnonFtzr, BookEndsFtzr, CanGram, DefaultAscii, DefaultUnicode,
//    Doc, DocFtzr, EmptyFtzr, FeatEntry, Featurizer, FuzzyEntry, MultiFtzr, SkipScheme,
//};
mod clustering;
use clustering::{cluster, cluster_B};

pub mod metrics;
use metrics::{Counted, Hamming, Jaccard, Metric, OverlapCoefficient};

mod dualiter;
use dualiter::*;

pub mod ftzrs;
//use ftzrs::{book_ends, n_gram, skipgram, CanGram, EmptyFtzr, Mapped, MultiFtzr};
use ftzrs::*;

pub mod fuzzyindex;
use fuzzyindex::{FuzzyIndex, ScratchPad, SearchParams};

pub mod fuzzypoint;
use fuzzypoint::{FromFeatures, FuzzyPoint, Labeled, SimHash, SimplePoint};

pub mod hasfeatures;
use hasfeatures::{HasFeatures, HasLabel};

mod utils;
use utils::{get_entry, open_lexicon, rec_rev_str, shuffle, Entry};
//fn test_index<Ftzr: CanGram, Point: FuzzyPoint>(lookup: &FuzzyIndex<&String, Ftzr, Point>) {

fn induce_typo(word: &str) -> String {
    let mut messed_up: String = "A".to_owned();
    messed_up.push_str(&unidecode(&word));

    if word.len() > 4 {
        messed_up.insert_str(4, "E");
    };
    messed_up
}

#[macro_export]
macro_rules! featurizers {
    () => {
        (EmptyFtzr)
    };
    ($a:expr $(, $tail:expr)*) => {{
        MultiFtzr {
            a: $a,
            b: featurizers!($($tail), *),
        }
    }};
}

macro_rules! record_stats (
    ($sp_name:ident, $action:ident, $val:expr) => {
        $sp_name.stats.as_mut().map(|x| x.$action($val));
    };

//    ($sp_name:ident, $action:ident) => {
//        $sp_name.stats.as_mut().map(StatsPad::$action);
//   };
);
