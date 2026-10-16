//! Reusable owned storage for discarded residue coefficients and masks.

pub(crate) struct VecPool<T> {
    buffers: Vec<Vec<T>>,
}

impl<T> Default for VecPool<T> {
    fn default() -> Self {
        Self {
            buffers: Vec::new(),
        }
    }
}

impl<T> VecPool<T> {
    pub(crate) fn recycle(&mut self, buffer: Vec<T>) {
        self.buffers.push(buffer);
    }
}

impl<T: Clone> VecPool<T> {
    pub(crate) fn take_from_slice(&mut self, values: &[T]) -> Vec<T> {
        let mut buffer = self.buffers.pop().unwrap_or_default();
        buffer.truncate(values.len());
        let initialized = buffer.len();
        buffer.clone_from_slice(&values[..initialized]);
        buffer.extend_from_slice(&values[initialized..]);
        buffer
    }
}
