use std::cmp;
use std::collections::hash_map::DefaultHasher;
use std::collections::HashSet;
use std::fs::File;
use std::hash::{Hash, Hasher};
use std::io::prelude::*;
use std::path::Path;
use std::time::Instant;

#[derive(Copy, Clone, Debug)]
pub struct Entry<Id, T> {
    pub id: Id,
    pub entry: T,
}

pub fn get_entry<'a, Id: Ord, T>(
    entries: &'a [Entry<Id, T>],
    t: &Id,
) -> Option<(usize, &'a Entry<Id, T>)> {
    unsafe {
        match entries.binary_search_by(|e| e.id.cmp(t)) {
            Ok(idx) => Some((idx, entries.get_unchecked(idx))),
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

pub fn shuffle<T: Hash + Clone>(items: &[T]) -> Vec<T> {
    let mut s = DefaultHasher::new();
    let start = Instant::now();
    let mut make_hash = |_t: T| {
        start.elapsed().as_nanos().hash(&mut s);
        s.finish()
    };
    let mut xs: Vec<_> = Iterator::collect(items.iter().map(|t| Entry {
        id: make_hash(t.clone()),
        entry: t,
    }));
    xs.sort();
    Iterator::collect(xs.into_iter().map(|e| e.entry.clone()))
}

pub fn rec_rev_str(mut s: String) -> String {
    if s.is_empty() {
        s
    } else {
        let removed_char = s.remove(0);
        let mut s = rec_rev_str(s);
        s.push(removed_char);
        s
    }
}
////////////////////
///
pub fn open_lexicon(path: &Path) -> HashSet<String> {
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
    Iterator::collect(s.lines().map(|x| x.to_owned()))
}

fn palindrome_lexicon(lx: &mut HashSet<String>) {
    let v: Vec<String> = Iterator::collect(lx.iter().map(|s| rec_rev_str(s.to_owned())));
    lx.extend(v.into_iter());
}
