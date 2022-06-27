use parking_lot::Mutex;
use std::sync::Arc;

// std::borrow::Cow has too many type restrictions to use instead of this.
// There's probably something in the std library that does the same thing tho?
pub enum BorrowedOrOwned<'a, T> {
    Owned(T),
    Borrowed(&'a T),
}

impl<'a, T> BorrowedOrOwned<'a, T> {
    pub fn borrow(&'a self) -> &'a T {
        match self {
            Self::Owned(value) => &value,
            Self::Borrowed(reference) => reference,
        }
    }
}

pub struct Setter<T>(pub Arc<Mutex<Option<T>>>);

impl<T> Setter<T> {
    pub fn set(&self, value: T) {
        *self.0.lock() = Some(value);
    }
}

impl<T> Clone for Setter<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone())
    }
}

pub struct Swappable<T> {
    inner: T,
    pub setter: Setter<T>,
}

impl<T> Swappable<T> {
    pub fn new(value: T) -> Self {
        Self {
            inner: value,
            setter: Setter(Default::default()),
        }
    }

    pub fn get(&mut self) -> &T {
        if let Some(mut lock) = self.setter.0.try_lock() {
            if let Some(replacement) = lock.take() {
                self.inner = replacement;
            }
        }

        &self.inner
    }
}
