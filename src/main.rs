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

mod pointfactories;
use pointfactories::{HasFeatures, HasName};

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

//impl<T: Ord> MetricPoint for FuzzyEntry<T> {
//    fn distance(&self, rhs: &Self) -> u64 {
//        //space::f64_metric(1.0 - self.sim(rhs))
//        self.ham_dist(rhs)
//    }
//}

fn main() {
    let mut i = 0.0;
    let k: usize = !0;

    //return ();
    let path = Path::new("/home/logan/Downloads/us_cities_states_counties.csv");

    let display = path.display();

    // Open the path in read-only mode, returns `io::Result<File>`
    let mut file = match File::open(&path) {
        Err(why) => panic!("couldn't open {}: {}", display, why),
        Ok(file) => file,
    };

    // Read the file contents into a string, returns `io::Result<usize>`
    let mut s = String::new();
    match file.read_to_string(&mut s) {
        Err(why) => panic!("couldn't read {}: {}", display, why),
        Ok(_) => {} //print!("{} contains:\n{}", display, s),
    }
    s = s.to_uppercase();
    let mut words = HashSet::new();
    for line in s.lines() {
        let line_words: Vec<_> = line.split("|").collect();
        if line_words.len() > 2 {
            if true {
                //////////////////////////////////////////////////////////////////////////////////////
                words.insert(line_words[0].to_owned());
                words.insert(rec_rev_str(line_words[0].to_owned()));
            } else if line_words[1] == "MI" {
                words.insert(line_words[0].to_owned());
            }
        }
    }
    //book_ends((2, 2), n_gram(2)).run(&[], &mut |x| {});
    //let lookup = FuzzyIndex::new(DefaultAscii, words.iter().map(|a| a.to_owned()));

    /*
    Origin: Hash + Debug + Ord + Clone,
    Id: Hash + Debug + Ord + Clone,
    G_Id: Fn(Origin) -> Id,
    G_T: Fn(&Origin) -> &[Token],
    U_T: CanGram, */

    let ftzr = featurizers![skipgram(2, (0, 3), 2), book_ends((4, 4), n_gram(2))];
    //let ftzr = featurizers![n_gram(2), n_gram(3)];
    let lexicon: Vec<_> = Iterator::collect(
        open_lexicon(Path::new(
            "/home/logan/Dropbox/USUABLE/en_pl_lexemes/en.txt",
        ))
        .into_iter()
        .take(100_000),
    );

    let training_data: Vec<TrainingAtom<String>> =
        Iterator::collect(lexicon.iter().map(|p| TrainingAtom {
            correct: p.to_owned(),
            typo: induce_typo(&p),
        }));

    {
        let lookup: FuzzyIndex<String, _, SimHash> =
            FuzzyIndex::new(ftzr.clone(), lexicon.clone().into_iter());
        let params = SearchParams {
            metric: Hamming,
            depth: 100,
            breadth: 100,
            max_comparisons: 50,
            return_if_gt: 70,
            timeout_n: 25,
        };
        println!("\nSENOR_BORROSO: {:?}", params);
        let mut t1 = FuzzyIndexTest::new(lookup, params);
        t1.run(training_data.clone());
        /////////////////////////////////
        //println!("\nHNSW: Hamming, M=12, M0=24, ef=400");
        //let mut hnt: HnswTester<_, _, SimHash, 12, 24> =
        //    HnswTester::new(ftzr, Hamming, 400, lexicon.clone());
        //hnt.run(training_data.clone());
    }

    {
        let mut lookup: FuzzyIndex<String, _, SimplePoint> =
            FuzzyIndex::new(ftzr.clone(), lexicon.clone().into_iter());
        lookup.compress_index(10);

        let params = SearchParams {
            metric: Jaccard,
            depth: 20,
            breadth: 20,
            max_comparisons: 70,
            return_if_gt: 70,
            timeout_n: 20,
        };
        println!("\nSENOR_BORROSO: {:?}", params);
        let mut t1 = FuzzyIndexTest::new(lookup, params);
        t1.run(training_data.clone());
        return ();
        println!("\nHNSW: Jaccard, M=12, M0=24, ef=40");
        let mut hnt: HnswTester<_, _, SimplePoint, 12, 24> =
            HnswTester::new(ftzr, Jaccard, 40, lexicon);
        hnt.run(training_data);
    }
}
