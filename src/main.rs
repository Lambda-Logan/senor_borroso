use std::cmp;
use std::collections::HashMap;
use std::collections::HashSet;
use std::convert::Into;
use std::default::Default;
use std::fmt;
use std::fs::File;
use std::hash::{BuildHasherDefault, Hasher};
use std::io::prelude::*;
use std::iter::FromIterator;
use std::mem;
use std::path::Path;
use std::rc::Rc;
use std::str;
use std::time::Instant;
#[allow(missing_copy_implementations)]
struct FnvHasher(u64);

impl Default for FnvHasher {
    #[inline]
    fn default() -> FnvHasher {
        FnvHasher(0xcbf29ce484222325)
    }
}

impl Hasher for FnvHasher {
    #[inline]
    fn finish(&self) -> u64 {
        self.0
    }

    #[inline]
    fn write(&mut self, bytes: &[u8]) {
        let FnvHasher(mut hash) = *self;

        for byte in bytes.iter() {
            hash = hash ^ (*byte as u64);
            hash = hash.wrapping_mul(0x100000001b3);
        }

        *self = FnvHasher(hash);
    }
}

/// A builder for default FNV hashers.
type FnvBuildHasher = BuildHasherDefault<FnvHasher>;
type FnvHashMap<K, V> = HashMap<K, V, FnvBuildHasher>;

#[derive(Hash, Copy, Clone, PartialEq, Ord, PartialOrd, Eq)]
struct Tok(u8, [u8; 2]);

impl fmt::Debug for Tok {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let a_val = &[self.0];
        let a = str::from_utf8(a_val).unwrap();
        let bc = str::from_utf8(&self.1).unwrap();
        f.debug_tuple("Tok").field(&a).field(&bc).finish()
    }
}
#[derive(Copy, Clone, Debug)]
struct Entry<Id, T> {
    id: Id,
    entry: T,
}

fn get_entry<'a, Id: Ord, T>(entries: &'a [Entry<Id, T>], t: &Id) -> Option<&'a Entry<Id, T>> {
    unsafe {
        match entries.binary_search_by(|e| e.id.cmp(t)) {
            Ok(idx) => Some(entries.get_unchecked(idx)),
            _ => None,
        }
    }
}

impl<T, Id: PartialEq> PartialEq for Entry<Id, T> {
    fn eq(&self, other: &Self) -> bool {
        self.id == other.id
    }
}

impl<T, Id: Eq> Eq for Entry<Id, T> {}

impl<T, Id: PartialOrd> PartialOrd for Entry<Id, T> {
    fn partial_cmp(&self, other: &Self) -> Option<cmp::Ordering> {
        self.id.partial_cmp(&other.id)
    }
}

impl<T, Id: Ord> Ord for Entry<Id, T> {
    fn cmp(&self, other: &Self) -> cmp::Ordering {
        self.id.cmp(&other.id)
    }
}

fn update_gram<F>(s: &[u8], updt: &mut F)
where
    F: FnMut(Tok) -> (),
{
    let l = s.len();
    for c in 1..l {
        let b = c - 1;
        let _a = cmp::max(0, (b as isize) - 3) as usize;
        for a in _a..b {
            updt(Tok(s[a], [s[b], s[c]]));
        }
    }
}

#[derive(Copy, Clone, Debug)]
enum AndOrOr<T> {
    Left(T),
    Right(T),
    Both(T, T),
}

struct DualIter<L, T, R> {
    left: L,
    right: R,
    item: Option<AndOrOr<T>>,
}

impl<L, T, R> DualIter<L, T, R>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    fn new(mut l: L, mut r: R) -> DualIter<L, T, R> {
        let maybe_item = Self::next_left_right(&mut l, &mut r);
        DualIter {
            left: l,
            right: r,
            item: maybe_item,
        }
    }

    fn next_left_right(l: &mut L, r: &mut R) -> Option<AndOrOr<T>> {
        match l.next() {
            Some(x) => Some(AndOrOr::Left(x)),
            None => match r.next() {
                Some(x) => Some(AndOrOr::Left(x)),
                None => None,
            },
        }
    }
}

impl<L, T, R> Iterator for DualIter<L, T, R>
where
    T: Ord,
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    type Item = AndOrOr<T>;
    #[inline(always)]
    fn next(&mut self) -> Option<Self::Item> {
        let mut self_item = None;
        //let mut ret = None;
        mem::swap(&mut self_item, &mut self.item);
        match self_item {
            Some(AndOrOr::Left(l)) => match self.right.next() {
                None => {
                    return Some(AndOrOr::Left(l));
                }
                Some(r) => {
                    if l < r {
                        self.item = Some(AndOrOr::Right(r));
                        return Some(AndOrOr::Left(l));
                    } else if l == r {
                        self.item = DualIter::next_left_right(&mut self.left, &mut self.right);
                        return Some(AndOrOr::Both(l, r));
                    } else {
                        self.item = Some(AndOrOr::Left(l));
                        return Some(AndOrOr::Right(r));
                    }
                }
            },
            Some(AndOrOr::Right(r)) => match self.left.next() {
                None => {
                    return Some(AndOrOr::Right(r));
                }
                Some(l) => {
                    if l < r {
                        self.item = Some(AndOrOr::Right(r));
                        return Some(AndOrOr::Left(l));
                    } else if l == r {
                        self.item = DualIter::next_left_right(&mut self.left, &mut self.right);
                        return Some(AndOrOr::Both(l, r));
                    } else {
                        self.item = Some(AndOrOr::Left(l));
                        return Some(AndOrOr::Right(r));
                    }
                }
            },
            Some(AndOrOr::Both(l, r)) => {
                self.item = DualIter::next_left_right(&mut self.left, &mut self.right);
                return Some(AndOrOr::Both(l, r));
            }
            None => match DualIter::next_left_right(&mut self.left, &mut self.right) {
                Some(x) => {
                    self.item = Some(x);
                    return self.next();
                }
                None => {
                    return None;
                }
            },
        }
    }
}

#[derive(Debug, Clone, PartialOrd, Ord, PartialEq, Eq)]
struct FuzzyEntry {
    string: String,
    bow: Vec<Tok>, //FnvHashMap<Tok, u8>,
}

impl FuzzyEntry {
    fn new(s: String) -> FuzzyEntry {
        let mut vbow: Vec<Tok> = Vec::with_capacity(4);
        {
            let mut updt = |t: Tok| vbow.push(t);
            update_gram(&s.as_bytes(), &mut updt);
        }
        vbow.sort();
        let r = FuzzyEntry {
            string: s,
            bow: vbow,
        };
        //println!("\n{:?}\n", r.bow.len());
        r
    }

    #[inline(always)]
    fn sim(&self, other: &Self) -> f64 {
        let mut n = 0;
        let mut d = 0;
        for x in DualIter::new(self.bow.iter(), other.bow.iter()) {
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
        (n as f64 / d as f64) //.sqrt()
    }
}
#[derive(Clone, Debug)]
struct FuzzyLookup {
    words: Vec<FuzzyEntry>,
    toks: Vec<Entry<Tok, Vec<u32>>>,
}

impl FuzzyLookup {
    fn new<W: Iterator<Item = String>>(wrds: W) -> FuzzyLookup {
        let words_set: HashSet<String> = Iterator::collect(wrds);
        // sort words set???
        let mut words: Vec<FuzzyEntry> =
            Iterator::collect(words_set.into_iter().map(FuzzyEntry::new));
        words.sort();
        let mut toks_hm: HashMap<Tok, Vec<u32>> = HashMap::with_capacity(words.len());
        let empty = Vec::new();
        for (idx, word) in words.iter().enumerate() {
            for tok in word.bow.iter() {
                let v = toks_hm.entry(*tok).or_insert(empty.clone());
                v.push(idx as u32);
            }
        }
        let mut toks = Vec::with_capacity(toks_hm.len());

        for (tok, idxs) in toks_hm.into_iter() {
            //println!("{:?}", idxs.len());
            toks.push(Entry {
                id: tok,
                entry: idxs,
            });
        }
        toks.sort();
        //TODO randomize order of the Vec<u32>'s in toks??
        println!("w t {:?}", (words.len(), toks.len()));

        let r = FuzzyLookup {
            words: words,
            toks: toks,
        };
        //println!("{:?}", r);
        r
    }

    fn lookup(&self, word: String) -> Option<(String, f64)> {
        let mut word_info = FuzzyEntry::new(word);
        word_info.bow.sort_unstable_by(|a, b| {
            let a_freq = get_entry(&self.toks, &a)
                .map(|e| e.entry.len())
                .unwrap_or(0);
            let b_freq = get_entry(&self.toks, &b)
                .map(|e| e.entry.len())
                .unwrap_or(0);
            a_freq.cmp(&b_freq)
        });

        //println!("{:?}", word_info);
        let mut freqs: HashMap<u32, i32> = HashMap::new(); //with_hasher(FnvHasher::default());
        let mut n_times = 0;
        for tok in word_info.bow.iter() {
            if n_times > 10 {
                break;
            }
            match get_entry(&self.toks, &tok) {
                Some(e) => {
                    //println!("idx {:?}", idx);
                    for word_idx in e.entry.iter() {
                        *freqs.entry(*word_idx).or_insert(0) += 1;
                        n_times += 1;
                    }
                }
                _ => {}
            }
        }
        //println!("n_times {:?}", n_times);
        let n_toks = word_info.bow.len() as f64;
        let mut r: Option<(String, f64)> = None;
        let mut max_so_far: f64 = 0.0;
        //println!("{:?}", freqs);
        for (idx, freq) in freqs.into_iter() {
            //println!("{:?}", (idx, freq));
            let word_b_info = &self.words[idx as usize];
            let n_b_toks = word_b_info.bow.len() as f64;
            let jaccard = ((freq * 2) as f64) / (n_toks + n_b_toks);
            if jaccard.gt(&max_so_far) {
                r = Some((word_b_info.string.clone(), jaccard));
                max_so_far = jaccard;
            }
        }
        r
    }
}

fn main() {
    let words = vec![
        //"zabc".to_owned(),
        //"ab cd".to_owned(),
        "grand ".to_owned(),
        "grand traverse".to_owned(),
        "traverse city".to_owned(),
        "carson city".to_owned(),
        //"abcde".to_owned(),
        //"lmnop".to_owned(),
        //"xyz".to_owned(),
    ]
    .into_iter();
    //let lookup = FuzzyLookup::new(words);
    //println!("{:?}", lookup.lookup("gra".to_owned()));
    let mut i = 0.0;

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
                //line_words[1] == "MI" {
                words.insert(line_words[0].to_owned());
            }
        }
    }

    let lookup = FuzzyLookup::new(words.iter().map(|a| a.to_owned()));
    let mut wins = 0;
    let mut n_cities_done: usize = 0;
    let n_cities = 10000;

    let fe: Vec<_> = Iterator::collect(words.iter().map(|w| FuzzyEntry::new(w.to_owned())));
    let start = Instant::now();
    for c1 in fe.iter().take(1000) {
        for c2 in fe.iter().take(1000) {
            c1.sim(c2);
            n_cities_done += 1;
        }
    }

    for city in words.iter().take(n_cities) {
        break;
        n_cities_done += 1;
        let mut messed_up: String = "A".to_owned();
        messed_up.push_str(city);
        //let e = FuzzyEntry::new(messed_up);
        if false {
            match lookup.lookup(messed_up) {
                Some((r, _)) => {
                    if &r == city {
                        wins += 1;
                    }
                }
                _ => {}
            }
        }
    }
    let elapsed = start.elapsed().as_micros();
    println!(
        "TIME EACH: {:?}",
        elapsed / cmp::max(n_cities_done as u128, 1)
    );
    println!(
        "{:?}",
        (wins, n_cities_done, (wins as f32) / (n_cities_done as f32))
    );

    let l = vec![0, 0, 4, 14, 20, 20, 24];
    let r = vec![4, 10, 10, 14, 24];
    for i in DualIter::new(l.iter(), r.iter()) {
        println!("{:?}", i);
    }
    println!(
        "{:?}",
        FuzzyEntry::new("GRAN RAPIDS".to_owned()).sim(&FuzzyEntry::new("GRAND RAPIDS".to_owned()))
    )
}
