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

//! [`ReadPlan`] and [`ReadPlanBuilder`] for determining which rows to read
//! from a Parquet file

use crate::arrow::array_reader::ArrayReader;
use crate::arrow::arrow_reader::selection::RowSelectionPolicy;
use crate::arrow::arrow_reader::selection::RowSelectionStrategy;
use crate::arrow::arrow_reader::{
    ArrowPredicate, ParquetRecordBatchReader, RowSelection, RowSelectionCursor, RowSelector,
};
use crate::errors::{ParquetError, Result};
use arrow_array::{Array, BooleanArray};
use arrow_buffer::{BooleanBuffer, BooleanBufferBuilder};
use arrow_select::filter::prep_null_mask_filter;
use std::collections::VecDeque;

/// A builder for [`ReadPlan`]
#[derive(Clone, Debug)]
pub struct ReadPlanBuilder {
    batch_size: usize,
    /// Which rows to select. Includes the result of all filters applied so far
    selection: Option<RowSelection>,
    /// Policy to use when materializing the row selection
    row_selection_policy: RowSelectionPolicy,
}

impl ReadPlanBuilder {
    /// Create a `ReadPlanBuilder` with the given batch size
    pub fn new(batch_size: usize) -> Self {
        Self {
            batch_size,
            selection: None,
            row_selection_policy: RowSelectionPolicy::default(),
        }
    }

    /// Set the current selection to the given value
    pub fn with_selection(mut self, selection: Option<RowSelection>) -> Self {
        self.selection = selection;
        self
    }

    /// Configure the policy to use when materialising the [`RowSelection`]
    ///
    /// Defaults to [`RowSelectionPolicy::Auto`]
    pub fn with_row_selection_policy(mut self, policy: RowSelectionPolicy) -> Self {
        self.row_selection_policy = policy;
        self
    }

    /// Returns the current row selection policy
    pub fn row_selection_policy(&self) -> &RowSelectionPolicy {
        &self.row_selection_policy
    }

    /// Returns the current selection, if any
    pub fn selection(&self) -> Option<&RowSelection> {
        self.selection.as_ref()
    }

    /// Specifies the number of rows in the row group, before filtering is applied.
    ///
    /// Returns a [`LimitedReadPlanBuilder`] that can apply
    /// offset and limit.
    ///
    /// Call [`LimitedReadPlanBuilder::build_limited`] to apply the limits to this
    /// selection.
    pub(crate) fn limited(self, row_count: usize) -> LimitedReadPlanBuilder {
        LimitedReadPlanBuilder::new(self, row_count)
    }

    /// Returns true if the current plan selects any rows
    pub fn selects_any(&self) -> bool {
        self.selection
            .as_ref()
            .map(|s| s.selects_any())
            .unwrap_or(true)
    }

    /// Returns the number of rows selected, or `None` if all rows are selected.
    pub fn num_rows_selected(&self) -> Option<usize> {
        self.selection.as_ref().map(|s| s.row_count())
    }

    /// Returns the [`RowSelectionStrategy`] for this plan.
    ///
    /// Guarantees to return either `Selectors` or `Mask`, never `Auto`.
    pub(crate) fn resolve_selection_strategy(&self) -> RowSelectionStrategy {
        match self.row_selection_policy {
            RowSelectionPolicy::Selectors => RowSelectionStrategy::Selectors,
            RowSelectionPolicy::Mask => RowSelectionStrategy::Mask,
            RowSelectionPolicy::Auto { threshold, .. } => {
                let selection = match self.selection.as_ref() {
                    Some(selection) => selection,
                    None => return RowSelectionStrategy::Selectors,
                };

                // total_rows: total number of rows selected / skipped
                // effective_count: number of non-empty selectors
                let (total_rows, effective_count) =
                    selection.iter().fold((0usize, 0usize), |(rows, count), s| {
                        if s.row_count > 0 {
                            (rows + s.row_count, count + 1)
                        } else {
                            (rows, count)
                        }
                    });

                if effective_count == 0 {
                    return RowSelectionStrategy::Mask;
                }

                if total_rows < effective_count.saturating_mul(threshold) {
                    RowSelectionStrategy::Mask
                } else {
                    RowSelectionStrategy::Selectors
                }
            }
        }
    }

    /// Evaluates an [`ArrowPredicate`], updating this plan's `selection`
    ///
    /// If the current `selection` is `Some`, the resulting [`RowSelection`]
    /// will be the conjunction of the existing selection and the rows selected
    /// by `predicate`.
    ///
    /// Note: pre-existing selections may come from evaluating a previous predicate
    /// or if the [`ParquetRecordBatchReader`] specified an explicit
    /// [`RowSelection`] in addition to one or more predicates.
    pub fn with_predicate(
        mut self,
        array_reader: Box<dyn ArrayReader>,
        predicate: &mut dyn ArrowPredicate,
    ) -> Result<Self> {
        let reader = ParquetRecordBatchReader::new(array_reader, self.clone().build());
        let mut filters = vec![];
        for maybe_batch in reader {
            let maybe_batch = maybe_batch?;
            let input_rows = maybe_batch.num_rows();
            let filter = predicate.evaluate(maybe_batch)?;
            // Since user supplied predicate, check error here to catch bugs quickly
            if filter.len() != input_rows {
                return Err(arrow_err!(
                    "ArrowPredicate predicate returned {} rows, expected {input_rows}",
                    filter.len()
                ));
            }
            match filter.null_count() {
                0 => filters.push(filter),
                _ => filters.push(prep_null_mask_filter(&filter)),
            };
        }

        // If the predicate selected all rows and there is no prior selection,
        // skip creating a RowSelection entirely — this avoids the allocation
        // and keeps selection as None which enables coalesced page fetches.
        let all_selected = filters.iter().all(|f| f.true_count() == f.len());
        if all_selected && self.selection.is_none() {
            return Ok(self);
        }
        let raw = RowSelection::from_filters(&filters);
        self.selection = match self.selection.take() {
            Some(selection) => Some(selection.and_then(&raw)),
            None => Some(raw),
        };
        Ok(self)
    }

    /// Evaluate a sequence of [`ArrowPredicate`]s in lockstep, per-batch,
    /// updating this plan's `selection` with their conjunction.
    ///
    /// Unlike repeated calls to [`Self::with_predicate`] — which evaluate each
    /// predicate across the **entire** current row-group selection before
    /// moving to the next — this method pulls a batch from every predicate's
    /// reader, evaluates every predicate on that batch, and ANDs the results.
    ///
    /// The crucial difference is the behaviour when `limit` is `Some(L)`:
    /// match counts are accumulated **across the AND-combined filters**, and
    /// the loop stops pulling further batches as soon as `L` rows have
    /// survived the full predicate chain. Regardless of how the caller ordered
    /// the predicates, the most expensive predicate therefore stops
    /// processing once the combined filter has produced enough rows to
    /// satisfy the downstream limit.
    ///
    /// `array_readers` must be the same length as `predicates`, with
    /// `array_readers[i]` projecting exactly the columns required by
    /// `predicates[i]`. Each reader is driven from a fresh clone of the
    /// current plan so the readers advance in lockstep over the same row
    /// positions.
    ///
    /// `total_rows` is the total row count of the row group (i.e. the row
    /// count that any existing selection on this builder covers, whether via
    /// `select` or `skip`). When the limit short-circuits before the last
    /// batch, we have only produced filters for the rows we actually
    /// processed, and the remaining rows of the row group must be appended
    /// to the result selection as skipped so `RowSelection::and_then` stays
    /// aligned with the outer selection's total row count.
    ///
    /// Compared to [`Self::with_predicate`], this method decodes all
    /// predicate columns for every batch it processes, rather than letting an
    /// earlier predicate's selection narrow later predicates' decode. That is
    /// only a net win when the combined-limit short-circuit keeps the number
    /// of processed batches small — typically when there is a tight `LIMIT`
    /// alongside an `ORDER BY` that preserves file order.
    pub fn with_predicates_limited(
        mut self,
        array_readers: Vec<Box<dyn ArrayReader>>,
        predicates: &mut [&mut dyn ArrowPredicate],
        limit: Option<usize>,
        total_rows: usize,
    ) -> Result<Self> {
        if predicates.is_empty() {
            return Ok(self);
        }
        if array_readers.len() != predicates.len() {
            return Err(general_err!(
                "with_predicates_limited: array_readers len ({}) != predicates len ({})",
                array_readers.len(),
                predicates.len()
            ));
        }

        // Each reader needs its own ReadPlan, but driven from the same
        // current selection so the per-batch row positions line up.
        let mut readers: Vec<ParquetRecordBatchReader> = array_readers
            .into_iter()
            .map(|ar| ParquetRecordBatchReader::new(ar, self.clone().build()))
            .collect();

        let mut filters: Vec<BooleanArray> = vec![];
        let mut cumulative: usize = 0;
        let mut processed_rows: usize = 0;
        let mut stopped_early = false;

        loop {
            // Pull one batch from every reader.
            let mut batches = Vec::with_capacity(readers.len());
            let mut ended = false;
            for r in readers.iter_mut() {
                match r.next() {
                    None => {
                        ended = true;
                        break;
                    }
                    Some(res) => batches.push(res?),
                }
            }
            if ended {
                // Assume all readers finish together: they share the same
                // read plan and batch size. Any drift would surface below
                // as a batch-length mismatch.
                break;
            }

            let batch_len = batches[0].num_rows();
            for b in &batches {
                if b.num_rows() != batch_len {
                    return Err(general_err!(
                        "with_predicates_limited: predicate readers desynchronized (batch sizes {} vs {})",
                        batch_len,
                        b.num_rows()
                    ));
                }
            }

            // Evaluate each predicate on its own batch, AND the results.
            let mut combined_buf: Option<BooleanBuffer> = None;
            for (pred, batch) in predicates.iter_mut().zip(batches.into_iter()) {
                let f = pred.evaluate(batch)?;
                if f.len() != batch_len {
                    return Err(general_err!(
                        "ArrowPredicate predicate returned {} rows, expected {}",
                        f.len(),
                        batch_len
                    ));
                }
                let f = match f.null_count() {
                    0 => f,
                    _ => prep_null_mask_filter(&f),
                };
                combined_buf = Some(match combined_buf {
                    None => f.values().clone(),
                    Some(prev) => &prev & f.values(),
                });
            }
            let combined = BooleanArray::new(combined_buf.unwrap(), None);
            let batch_matches = combined.true_count();

            processed_rows += batch_len;
            match limit {
                Some(lim) if cumulative + batch_matches >= lim => {
                    let needed = lim - cumulative;
                    filters.push(truncate_filter_after_n_trues(&combined, needed));
                    stopped_early = true;
                    break;
                }
                _ => {
                    cumulative += batch_matches;
                    filters.push(combined);
                }
            }
        }

        // If we did not stop early, every batch was fully selected, and there
        // is no prior selection, skip materialising a RowSelection.
        if !stopped_early
            && self.selection.is_none()
            && filters.iter().all(|f| f.true_count() == f.len())
        {
            return Ok(self);
        }

        // `RowSelection::from_filters` produces a selection whose total row
        // count equals the number of rows we actually processed. If we
        // stopped early, pad with an explicit skip so the selection covers
        // the expected total (the outer selected row count when an outer
        // selection exists, else the full row group).
        //
        // `and_then`'s invariant is that `other.total == self.row_count()`
        // (number of **selected** rows in the outer). When there is no
        // outer selection, the resulting `raw` is used as-is, so the
        // expected total is the row group row count. Overshooting either
        // direction panics in `and_then`.
        let expected_total = match self.selection.as_ref() {
            Some(s) => s.row_count(),
            None => total_rows,
        };
        let mut raw = RowSelection::from_filters(&filters);
        if processed_rows < expected_total {
            let skip_tail = expected_total - processed_rows;
            let mut selectors: Vec<RowSelector> = raw.into();
            selectors.push(RowSelector::skip(skip_tail));
            raw = RowSelection::from(selectors);
        }

        self.selection = match self.selection.take() {
            Some(selection) => Some(selection.and_then(&raw)),
            None => Some(raw),
        };
        Ok(self)
    }

    /// Create a final `ReadPlan` the read plan for the scan
    pub fn build(mut self) -> ReadPlan {
        // If selection is empty, truncate
        if !self.selects_any() {
            self.selection = Some(RowSelection::from(vec![]));
        }

        // Preferred strategy must not be Auto
        let selection_strategy = self.resolve_selection_strategy();

        let Self {
            batch_size,
            selection,
            row_selection_policy: _,
        } = self;

        let selection = selection.map(|s| s.trim());

        let row_selection_cursor = selection
            .map(|s| {
                let trimmed = s.trim();
                let selectors: Vec<RowSelector> = trimmed.into();
                match selection_strategy {
                    RowSelectionStrategy::Mask => {
                        RowSelectionCursor::new_mask_from_selectors(selectors)
                    }
                    RowSelectionStrategy::Selectors => RowSelectionCursor::new_selectors(selectors),
                }
            })
            .unwrap_or(RowSelectionCursor::new_all());

        ReadPlan {
            batch_size,
            row_selection_cursor,
        }
    }
}

/// Builder for [`ReadPlan`] that applies a limit and offset to the read plan
///
/// See [`ReadPlanBuilder::limited`] to create this builder.
pub(crate) struct LimitedReadPlanBuilder {
    /// The underlying builder
    inner: ReadPlanBuilder,
    /// Total number of rows in the row group before the selection, limit or
    /// offset are applied
    row_count: usize,
    /// The offset to apply, if any
    offset: Option<usize>,
    /// The limit to apply, if any
    limit: Option<usize>,
}

impl LimitedReadPlanBuilder {
    /// Create a new `LimitedReadPlanBuilder` from the existing builder and number of rows
    fn new(inner: ReadPlanBuilder, row_count: usize) -> Self {
        Self {
            inner,
            row_count,
            offset: None,
            limit: None,
        }
    }

    /// Set the offset to apply to the read plan
    pub(crate) fn with_offset(mut self, offset: Option<usize>) -> Self {
        self.offset = offset;
        self
    }

    /// Set the limit to apply to the read plan
    pub(crate) fn with_limit(mut self, limit: Option<usize>) -> Self {
        self.limit = limit;
        self
    }

    /// Apply offset and limit, updating the selection on the underlying builder
    /// and returning it.
    pub(crate) fn build_limited(self) -> ReadPlanBuilder {
        let Self {
            mut inner,
            row_count,
            offset,
            limit,
        } = self;

        // If the selection is empty, truncate
        if !inner.selects_any() {
            inner.selection = Some(RowSelection::from(vec![]));
        }

        // If an offset is defined, apply it to the `selection`
        if let Some(offset) = offset {
            inner.selection = Some(match row_count.checked_sub(offset) {
                None => RowSelection::from(vec![]),
                Some(remaining) => inner
                    .selection
                    .map(|selection| selection.offset(offset))
                    .unwrap_or_else(|| {
                        RowSelection::from(vec![
                            RowSelector::skip(offset),
                            RowSelector::select(remaining),
                        ])
                    }),
            });
        }

        // If a limit is defined, apply it to the final `selection`
        if let Some(limit) = limit {
            inner.selection = Some(
                inner
                    .selection
                    .map(|selection| selection.limit(limit))
                    .unwrap_or_else(|| {
                        RowSelection::from(vec![RowSelector::select(limit.min(row_count))])
                    }),
            );
        }

        inner
    }
}

/// A plan reading specific rows from a Parquet Row Group.
///
/// See [`ReadPlanBuilder`] to create `ReadPlan`s
#[derive(Debug)]
pub struct ReadPlan {
    /// The number of rows to read in each batch
    batch_size: usize,
    /// Row ranges to be selected from the data source
    row_selection_cursor: RowSelectionCursor,
}

impl ReadPlan {
    /// Returns a mutable reference to the selection selectors, if any
    #[deprecated(since = "57.1.0", note = "Use `row_selection_cursor_mut` instead")]
    pub fn selection_mut(&mut self) -> Option<&mut VecDeque<RowSelector>> {
        if let RowSelectionCursor::Selectors(selectors_cursor) = &mut self.row_selection_cursor {
            Some(selectors_cursor.selectors_mut())
        } else {
            None
        }
    }

    /// Returns a mutable reference to the row selection cursor
    pub fn row_selection_cursor_mut(&mut self) -> &mut RowSelectionCursor {
        &mut self.row_selection_cursor
    }

    /// Return the number of rows to read in each output batch
    #[inline(always)]
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }
}

/// Return a `BooleanArray` with the same length as `filter` that keeps only
/// the first `n` `true` positions from `filter` (in order) and replaces every
/// other position with `false`.
///
/// This is used by [`ReadPlanBuilder::with_predicates_limited`] when a
/// combined filter's `true_count()` would otherwise push the cumulative match
/// total past a caller-supplied limit. Preserving the length matters because
/// the filter still needs to feed [`RowSelection::from_filters`] alongside
/// filters from earlier batches whose total rows equal the selection length.
fn truncate_filter_after_n_trues(filter: &BooleanArray, n: usize) -> BooleanArray {
    let len = filter.len();
    let mut builder = BooleanBufferBuilder::new(len);
    let mut kept = 0usize;
    for i in 0..len {
        // filter has no nulls at this point (callers pass `prep_null_mask_filter`-ed inputs
        // or freshly-constructed arrays with `None` nulls), so `value` is safe.
        if filter.value(i) && kept < n {
            builder.append(true);
            kept += 1;
        } else {
            builder.append(false);
        }
    }
    BooleanArray::new(builder.finish(), None)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn builder_with_selection(selection: RowSelection) -> ReadPlanBuilder {
        ReadPlanBuilder::new(1024).with_selection(Some(selection))
    }

    #[test]
    fn preferred_selection_strategy_prefers_mask_by_default() {
        let selection = RowSelection::from(vec![RowSelector::select(8)]);
        let builder = builder_with_selection(selection);
        assert_eq!(
            builder.resolve_selection_strategy(),
            RowSelectionStrategy::Mask
        );
    }

    #[test]
    fn preferred_selection_strategy_prefers_selectors_when_threshold_small() {
        let selection = RowSelection::from(vec![RowSelector::select(8)]);
        let builder = builder_with_selection(selection)
            .with_row_selection_policy(RowSelectionPolicy::Auto { threshold: 1 });
        assert_eq!(
            builder.resolve_selection_strategy(),
            RowSelectionStrategy::Selectors
        );
    }

    #[test]
    fn truncate_filter_after_n_trues_basic() {
        let filter = BooleanArray::from(vec![true, false, true, true, false, true]);
        let truncated = truncate_filter_after_n_trues(&filter, 2);
        assert_eq!(truncated.len(), filter.len());
        // first two `true`s preserved, rest false
        assert_eq!(
            truncated.iter().collect::<Vec<_>>(),
            vec![
                Some(true),
                Some(false),
                Some(true),
                Some(false),
                Some(false),
                Some(false)
            ]
        );
    }

    #[test]
    fn truncate_filter_after_n_trues_zero_keeps_none() {
        let filter = BooleanArray::from(vec![true, true, true]);
        let truncated = truncate_filter_after_n_trues(&filter, 0);
        assert_eq!(truncated.true_count(), 0);
        assert_eq!(truncated.len(), 3);
    }

    #[test]
    fn truncate_filter_after_n_trues_more_than_available() {
        let filter = BooleanArray::from(vec![true, false, true]);
        let truncated = truncate_filter_after_n_trues(&filter, 10);
        // Only two `true`s exist, cap is higher — keep everything that was true
        assert_eq!(truncated.true_count(), 2);
        assert_eq!(truncated.len(), 3);
    }
}
