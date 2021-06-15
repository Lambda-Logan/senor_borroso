use std::mem;

#[derive(Copy, Clone, Debug)]
pub enum AndOrOr<T> {
    Left(T),
    Right(T),
    Both(T, T),
}

pub struct DualIter<L, T, R> {
    left: L,
    right: R,
    item: Option<AndOrOr<T>>,
}

impl<L, T, R> DualIter<L, T, R>
where
    L: Iterator<Item = T>,
    R: Iterator<Item = T>,
{
    pub fn new(mut l: L, mut r: R) -> DualIter<L, T, R> {
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
