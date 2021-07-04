//#![allow(warnings)]
use std::collections::HashSet;
use std::fs::File;
use std::io::prelude::*;
use std::path::Path;
use std::str;
use unidecode::unidecode;

mod testing;
use testing::{FuzzyIndexTest, HnswTester, Testable, TrainingAtom};

mod feat;
//use feat::{
//    book_ends, n_gram, skipgram, AnonFtzr, BookEndsFtzr, CanGram, DefaultAscii, DefaultUnicode,
//    Doc, DocFtzr, EmptyFtzr, FeatEntry, Featurizer, FuzzyEntry, MultiFtzr, SkipScheme,
//};

mod dualiter;
use dualiter::*;

mod ftzrs;
use ftzrs::{book_ends, n_gram, skipgram, CanGram, EmptyFtzr, MultiFtzr};

mod fuzzyindex;
use fuzzyindex::{FuzzyIndex, SearchParams};

mod fuzzypoint;
use fuzzypoint::{Counted, FuzzyPoint, Hamming, Jaccard, Labeled, Metric, SimHash, SimplePoint};

mod hasfeatures;
use hasfeatures::{HasFeatures, HasName};

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

fn main() {}
