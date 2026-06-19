// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use super::{RowSelection, RowSelector};
use crate::errors::ParquetError;
use arrow_array::BooleanArray;
use arrow_buffer::bit_iterator::BitSliceIterator;
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};

/// Streaming RLE view of a [`BooleanBuffer`].
#[derive(Debug)]
pub struct MaskRunIter<'a> {
    slices: BitSliceIterator<'a>,
    cursor: usize,
    total: usize,
    pending: Option<RowSelector>,
    finished: bool,
}

impl<'a> MaskRunIter<'a> {
    pub(super) fn new(mask: &'a BooleanBuffer) -> Self {
        Self {
            slices: mask.set_slices(),
            cursor: 0,
            total: mask.len(),
            pending: None,
            finished: false,
        }
    }
}

impl Iterator for MaskRunIter<'_> {
    type Item = RowSelector;

    fn next(&mut self) -> Option<RowSelector> {
        if let Some(p) = self.pending.take() {
            return Some(p);
        }
        if self.finished {
            return None;
        }
        match self.slices.next() {
            Some((start, end)) => {
                let select = RowSelector::select(end - start);
                if start > self.cursor {
                    let skip = RowSelector::skip(start - self.cursor);
                    self.pending = Some(select);
                    self.cursor = end;
                    Some(skip)
                } else {
                    self.cursor = end;
                    Some(select)
                }
            }
            None => {
                self.finished = true;
                if self.cursor < self.total {
                    let skip = RowSelector::skip(self.total - self.cursor);
                    self.cursor = self.total;
                    Some(skip)
                } else {
                    None
                }
            }
        }
    }
}

/// Materialize a [`BooleanBuffer`] into its RLE form.
pub(crate) fn mask_to_selectors(mask: &BooleanBuffer) -> Vec<RowSelector> {
    let total_rows = mask.len();
    if total_rows == 0 {
        return Vec::new();
    }
    let mut selectors: Vec<RowSelector> = Vec::new();
    let mut last_end = 0;
    for (start, end) in mask.set_slices() {
        if start > last_end {
            selectors.push(RowSelector::skip(start - last_end));
        }
        selectors.push(RowSelector::select(end - start));
        last_end = end;
    }
    if last_end != total_rows {
        selectors.push(RowSelector::skip(total_rows - last_end));
    }
    selectors
}

/// Bitwise AND of two mask-backed selections. Longer side's tail passes through.
pub(super) fn intersect_masks(l: &BooleanBuffer, r: &BooleanBuffer) -> BooleanBuffer {
    if l.len() == r.len() {
        return l & r;
    }
    let common = l.len().min(r.len());
    let head = &l.slice(0, common) & &r.slice(0, common);
    let (longer, longer_len) = if l.len() > r.len() {
        (l, l.len())
    } else {
        (r, r.len())
    };
    let tail = longer.slice(common, longer_len - common);
    let mut builder = BooleanBufferBuilder::new(longer_len);
    builder.append_buffer(&head);
    builder.append_buffer(&tail);
    builder.finish()
}

/// Bitwise OR of two mask-backed selections. Longer side's tail passes through.
pub(super) fn union_masks(l: &BooleanBuffer, r: &BooleanBuffer) -> BooleanBuffer {
    if l.len() == r.len() {
        return l | r;
    }
    let common = l.len().min(r.len());
    let head = &l.slice(0, common) | &r.slice(0, common);
    let (longer, longer_len) = if l.len() > r.len() {
        (l, l.len())
    } else {
        (r, r.len())
    };
    let tail = longer.slice(common, longer_len - common);
    let mut builder = BooleanBufferBuilder::new(longer_len);
    builder.append_buffer(&head);
    builder.append_buffer(&tail);
    builder.finish()
}

/// Applies `other` to the selected rows of `mask`, preserving the original row domain.
pub(super) fn and_then_mask(mask: &BooleanBuffer, other: &RowSelection) -> BooleanBuffer {
    let mut builder = BooleanBufferBuilder::new(mask.len());
    let mut other_iter = other.iter();
    let mut current = other_iter.next();
    let mut cursor = 0usize;

    // Iterate only over the set positions in `mask`; the gaps of unset bits
    // are filled in bulk with `append_n` instead of bit-by-bit.
    for set_idx in mask.set_indices() {
        if set_idx > cursor {
            builder.append_n(set_idx - cursor, false);
        }
        cursor = set_idx + 1;

        while current.as_ref().is_some_and(|s| s.row_count == 0) {
            current = other_iter.next();
        }
        let selector = current
            .as_mut()
            .expect("selection contains less than the number of selected rows");
        let selected = !selector.skip;
        selector.row_count -= 1;
        builder.append(selected);
    }
    if cursor < mask.len() {
        builder.append_n(mask.len() - cursor, false);
    }

    if current.is_some_and(|s| s.row_count != 0) || other_iter.any(|s| s.row_count != 0) {
        panic!("selection exceeds the number of selected rows");
    }

    builder.finish()
}

/// Skips the first `offset` selected rows of a mask-backed selection.
pub(super) fn offset_mask(mask: BooleanBuffer, offset: usize) -> BooleanBuffer {
    let popcount = mask.count_set_bits();
    if offset >= popcount {
        return BooleanBuffer::new_unset(0);
    }
    // Position one past the `offset`-th set bit, i.e. the index of the first
    // selected row to keep.
    let pos = mask.find_nth_set_bit_position(0, offset);
    let mut builder = BooleanBufferBuilder::new(mask.len());
    builder.append_n(pos, false);
    builder.append_buffer(&mask.slice(pos, mask.len() - pos));
    builder.finish()
}

/// Keeps only the first `limit` selected rows of a mask-backed selection.
pub(super) fn limit_mask(mask: BooleanBuffer, limit: usize) -> BooleanBuffer {
    // `find_nth_set_bit_position` returns `mask.len()` when there are fewer
    // than `limit` set bits, so the slice naturally degrades to the original
    // mask in that case.
    let cut = mask.find_nth_set_bit_position(0, limit);
    mask.slice(0, cut)
}

/// Cursor for iterating a mask-backed [`RowSelection`]
///
/// This is best for dense selections where there are many small skips
/// or selections. For example, selecting every other row.
#[derive(Debug)]
pub struct MaskCursor {
    pub(super) mask: BooleanBuffer,
    /// Current absolute offset into the selection
    pub(super) position: usize,
}

impl MaskCursor {
    /// Returns `true` when no further rows remain
    pub fn is_empty(&self) -> bool {
        self.position >= self.mask.len()
    }

    /// Advance through the mask representation, producing the next chunk summary
    pub fn next_mask_chunk(&mut self, batch_size: usize) -> Option<MaskChunk> {
        let (initial_skip, chunk_rows, selected_rows, mask_start, end_position) = {
            let mask = &self.mask;

            if self.position >= mask.len() {
                return None;
            }

            let start_position = self.position;
            let mut cursor = start_position;
            let mut initial_skip = 0;

            while cursor < mask.len() && !mask.value(cursor) {
                initial_skip += 1;
                cursor += 1;
            }

            let mask_start = cursor;
            let mut chunk_rows = 0;
            let mut selected_rows = 0;

            // Advance until enough rows have been selected to satisfy the batch size,
            // or until the mask is exhausted. This mirrors the behaviour of the legacy
            // `RowSelector` queue-based iteration.
            while cursor < mask.len() && selected_rows < batch_size {
                chunk_rows += 1;
                if mask.value(cursor) {
                    selected_rows += 1;
                }
                cursor += 1;
            }

            (initial_skip, chunk_rows, selected_rows, mask_start, cursor)
        };

        self.position = end_position;

        Some(MaskChunk {
            initial_skip,
            chunk_rows,
            selected_rows,
            mask_start,
        })
    }

    /// Materialise the boolean values for a mask-backed chunk
    pub fn mask_values_for(&self, chunk: &MaskChunk) -> Result<BooleanArray, ParquetError> {
        if chunk.mask_start.saturating_add(chunk.chunk_rows) > self.mask.len() {
            return Err(ParquetError::General(
                "Internal Error: MaskChunk exceeds mask length".to_string(),
            ));
        }
        Ok(BooleanArray::from(
            self.mask.slice(chunk.mask_start, chunk.chunk_rows),
        ))
    }
}

/// Result of computing the next chunk to read when using a [`MaskCursor`]
#[derive(Debug)]
pub struct MaskChunk {
    /// Number of leading rows to skip before reaching selected rows
    pub initial_skip: usize,
    /// Total rows covered by this chunk (selected + skipped)
    pub chunk_rows: usize,
    /// Rows actually selected within the chunk
    pub selected_rows: usize,
    /// Starting offset within the mask where the chunk begins
    pub mask_start: usize,
}

pub(super) fn boolean_mask_from_selectors(selectors: &[RowSelector]) -> BooleanBuffer {
    let total_rows: usize = selectors.iter().map(|s| s.row_count).sum();
    let mut builder = BooleanBufferBuilder::new(total_rows);
    for selector in selectors {
        builder.append_n(selector.row_count, !selector.skip);
    }
    builder.finish()
}
