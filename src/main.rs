#![allow(warnings)]
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::prelude::*;
use std::io::prelude::*;
use std::path::Path;
use std::str;
use std::time::Instant;
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

mod metrics;
use metrics::{Counted, Hamming, Jaccard, Metric, OverlapCoefficient};

mod dualiter;
use dualiter::*;

mod ftzrs;
//use ftzrs::{book_ends, n_gram, skipgram, CanGram, EmptyFtzr, Mapped, MultiFtzr};
use ftzrs::*;

mod fuzzyindex;
use fuzzyindex::{FuzzyIndex, ScratchPad, SearchParams};

mod fuzzypoint;
use fuzzypoint::{FromFeatures, FuzzyPoint, Labeled, SimHash, SimplePoint};

mod hasfeatures;
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

fn main() {
    return ();
    let ftzr = book_ends((3, 3), n_gram(2));

    let v: Vec<String> = ftzr.featurize("Hello world!");

    println!("{:?}", v);

    return ();
    let mut doc = String::new();
    File::open(&Path::new(
        &"/home/logan/Dropbox/TESTING/scala/borroso/src/main/scala/example/test.txt",
    ))
    .expect(&"err opening file")
    .read_to_string(&mut doc)
    .unwrap();
    let vdoc: Vec<&str> = doc.split_ascii_whitespace().collect();
    let t = Instant::now();
    let epochs = 1000;
    for _ in (0..epochs) {
        let v: Vec<HashedFeature64> = ftzr.featurize(&vdoc);
    }
    println!("{:?}", t.elapsed().as_micros());
    /* let ftzr = book_ends((5, 5), n_gram(3)); // skipgram(2, (0, 3), 2);
                                             //let v = "This is my string";
                                             //let v: Vec<char> = Iterator::collect("one fish two fish red fish blue fish".chars());
    let s = "one fish TWO fish red fish blue fish";

    let val: (Vec<HashedFeature64>, HashMap<String, u8>, Vec<String>) = ftzr.featurize_x3(s);

    //let val: HashMap<HashedFeature64, u128> = ftzr.featurize(&v);
    println!("{:?}", val);
    return ();
    /// define a featurizer, in this case we use multiple with the 'featurizers' macro
    let ftzr = featurizers![skipgram(2, (0, 3), 2), book_ends((4, 4), n_gram(2))];
    //let ftzr = featurizers![n_gram(2), n_gram(3), n_gram(4), n_gram(5), n_gram(6)];

    let lexicon: Vec<_> = Iterator::collect(
        open_lexicon(Path::new(
            "/home/logan/Dropbox/USUABLE/en_pl_lexemes/en.txt",
        ))
        .into_iter()
        .take(10_000), //CHANGE TO 100_000 FOR FULL DATA
    );

    let training_data: Vec<TrainingAtom<String>> =
        Iterator::collect(lexicon.iter().map(|p| TrainingAtom {
            correct: p.to_owned(),
            typo: induce_typo(&p),
        }));

    //////////////////////////
    /// TESTING SENOR_BORROSO
    //////////////////////////
    //let mut fuzzyfind0: FuzzyIndex<String, _, SimHash> =
    //    FuzzyIndex::new(ftzr.clone(), lexicon.clone().into_iter());
    //fuzzyfind0.compress_index(10);
    let params = SearchParams {
        metric: Hamming,
        depth: 1000,
        breadth: 100,
        max_comparisons: 500,
        return_if_gt: 70,
        use_best_after: 250,
    };
    //println!("\nSENOR_BORROSO: {:?}", params);
    //let mut t1 = FuzzyIndexTest::new(fuzzyfind0, params);
    //t1.run(training_data.clone());

    let mut fuzzyfind_pre: FuzzyIndex<String, _, SimplePoint> =
        FuzzyIndex::new(ftzr.clone(), lexicon.clone().into_iter());

    //println!("TOTAL FEATURES: {:?}", fuzzyfind_pre.all_feats.len());
    let new_ftzr = Mapped {
        table: cluster_B(
            0.2,
            &fuzzyfind_pre.transpose(),
            &params.with_metric(OverlapCoefficient),
        ),
        //table: cluster(&mut fuzzyfind_pre.adj_matrix(), 0.7, 1),
        keep_unseen: true,
        ftzr: ftzr.clone(),
    };

    let v1 = SimplePoint::made_from(&"LOGAN", &ftzr);
    let v2 = SimplePoint::made_from(&"LOGAN", &new_ftzr);
    println!("{:?}", v1);
    println!("{:?}", v2);

    //fuzzyfind1.compress_index(10);

    let params = SearchParams {
        metric: Jaccard,
        depth: 5,
        breadth: 20,
        max_comparisons: 70,
        return_if_gt: 70,
        use_best_after: 20,
    };
    println!("P");
    let mut sp = ScratchPad::new();
    sp.stats = Some(Default::default());
    //for nbr in fuzzyfind1.neighbors_of((&"PLATIDEXTITUDE").to_string(), &mut sp, &params) {
    //println!("{:?}", nbr);
    //}
    println!("\nSENOR_BORROSO: {:?}", params);
    //record_stats!(sp, called, 100);

    let mut fuzzyfind1: FuzzyIndex<String, _, SimplePoint> =
        FuzzyIndex::new(ftzr, lexicon.clone().into_iter());
    println!("number of features: {:?}", fuzzyfind1.all_feats.len());
    let mut t1 = FuzzyIndexTest::new(fuzzyfind1, params, &mut sp);
    //sp.stats.map(StatsPad::inc_called(1));
    t1.run(training_data.clone());
    println!("{:?}", sp.stats.map(|s| s.average()));

    return ();
    //////////////////////////
    /// TESTING HNSW
    //////////////////////////
    println!("\nHNSW: Hamming, M=12, M0=24, ef=400");
    let mut hnsw0: HnswTester<_, _, SimHash, 12, 24> =
        HnswTester::new(ftzr, Hamming, 400, lexicon.clone());
    hnsw0.run(training_data.clone());

    println!("\nHNSW: Jaccard, M=12, M0=24, ef=40");
    let mut hnsw1: HnswTester<_, _, SimplePoint, 12, 24> =
        HnswTester::new(ftzr, Jaccard, 40, lexicon);
    hnsw1.run(training_data); */
}
