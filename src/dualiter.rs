use std::mem;

#[derive(Copy, Clone, Debug)]
pub(crate) enum AndOrOr<T> {
    Left(T),
    Right(T),
    Both(T, T),
}

pub(crate) struct DualIter<L, T, R> {
    left: L,
    right: R,
    item: Option<AndOrOr<T>>,
}

impl<L, T, R> DualIter<L, T, R>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    #[inline]
    pub(crate) fn new(mut l: L, mut r: R) -> DualIter<L, T, R> {
        let maybe_item = Self::next_left_right(&mut l, &mut r);
        DualIter {
            left: l,
            right: r,
            item: maybe_item,
        }
    }
    #[inline]
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

pub(crate) struct Uniq<Q, Item> {
    iter: Q,
    next: Option<Item>,
}

/////////////////
/// An iterator adaptor that merges consecutive elements that are equal
/// #EXAMPLE
/// ```
///     for i in Uniq::new(vec![9, 9, 2, 2, 5, 6, 6, 6, 6, 0, 1, 1, 0].iter()) {
///         println!("{:?}", i);
///     }
/// /// >>> 9, 2, 5, 6, 0, 1, 0
/// ```
////////////////
impl<Q, Item> Uniq<Q, Item> {
    #[inline]
    pub(crate) fn new(iter: Q) -> Self {
        Uniq {
            iter: iter,
            next: None,
        }
    }
}

impl<Q: Iterator<Item = Item>, Item: Eq> Iterator for Uniq<Q, Item> {
    type Item = Item;
    #[inline(always)]
    fn next(&mut self) -> Option<Item> {
        match &self.next {
            None => {
                let a = self.iter.next();
                let mut b = self.iter.next();
                while b.is_some() && b == a {
                    b = self.iter.next();
                }
                self.next = b;
                a
            }
            some_a => {
                let mut x = self.iter.next();
                while x.is_some() && &x == some_a {
                    x = self.iter.next();
                }
                mem::swap(&mut x, &mut self.next);
                x
            }
        }
    }
}
