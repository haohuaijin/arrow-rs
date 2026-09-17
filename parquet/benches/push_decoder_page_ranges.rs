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

//! Isolate buffer-range management when a row selection touches every data page.
//! Builds row-group readers without decoding pages. Each of eight columns has
//! 1, 10, 100 or 1,000 V1 pages; one row per page is selected.

use std::hint::black_box;
use std::sync::Arc;
use std::time::Duration;

use arrow_array::{ArrayRef, RecordBatch, UInt64Array};
use arrow_schema::{DataType, Field, Schema};
use bytes::Bytes;
use criterion::{BenchmarkId, Criterion, criterion_group, criterion_main};
use parquet::DecodeResult;
use parquet::arrow::ArrowWriter;
use parquet::arrow::arrow_reader::{
    ArrowReaderOptions, ParquetRecordBatchReaderBuilder, RowSelection, RowSelector,
};
use parquet::arrow::push_decoder::ParquetPushDecoderBuilder;
use parquet::file::metadata::{PageIndexPolicy, ParquetMetaData};
use parquet::file::properties::{EnabledStatistics, WriterProperties, WriterVersion};

fn dataset(pages: usize) -> (Bytes, Arc<ParquetMetaData>, RowSelection) {
    let rows = pages * 1_000;
    let schema = Arc::new(Schema::new(
        (0..8)
            .map(|i| Field::new(format!("c{i}"), DataType::UInt64, false))
            .collect::<Vec<_>>(),
    ));
    let values: ArrayRef = Arc::new(UInt64Array::from_iter_values(0..rows as u64));
    let batch = RecordBatch::try_new(Arc::clone(&schema), vec![values; 8]).unwrap();
    let props = WriterProperties::builder()
        .set_writer_version(WriterVersion::PARQUET_1_0)
        .set_dictionary_enabled(false)
        .set_statistics_enabled(EnabledStatistics::Chunk)
        .set_max_row_group_row_count(Some(rows))
        .set_data_page_row_count_limit(1_000)
        .set_write_batch_size(1_000)
        .build();
    let mut writer = ArrowWriter::try_new(Vec::new(), schema, Some(props)).unwrap();
    writer.write(&batch).unwrap();
    let data = Bytes::from(writer.into_inner().unwrap());
    let builder = ParquetRecordBatchReaderBuilder::try_new_with_options(
        data.clone(),
        ArrowReaderOptions::new().with_page_index_policy(PageIndexPolicy::Optional),
    )
    .unwrap();
    let metadata = Arc::clone(builder.metadata());
    assert_eq!(metadata.num_row_groups(), 1);
    assert_eq!(
        metadata
            .page_index_for_row_group(0)
            .offset_index(0)
            .unwrap()
            .page_locations()
            .len(),
        pages
    );
    let selection = RowSelection::from(
        (0..pages)
            .flat_map(|_| [RowSelector::skip(999), RowSelector::select(1)])
            .collect::<Vec<_>>(),
    );
    (data, metadata, selection)
}

fn build_reader(data: &Bytes, metadata: &Arc<ParquetMetaData>, selection: &RowSelection) {
    let mut decoder = ParquetPushDecoderBuilder::try_new_decoder(Arc::clone(metadata))
        .unwrap()
        .with_row_selection(selection.clone())
        .build()
        .unwrap();
    loop {
        match decoder.try_next_reader().unwrap() {
            DecodeResult::Data(reader) => {
                black_box(reader);
            }
            DecodeResult::Finished => break,
            DecodeResult::NeedsData(ranges) => {
                let buffers = ranges
                    .iter()
                    .map(|r| data.slice(r.start as usize..r.end as usize))
                    .collect();
                decoder.push_ranges(ranges, buffers).unwrap();
            }
        }
    }
}

fn bench_page_ranges(c: &mut Criterion) {
    let mut group = c.benchmark_group("push_decoder_page_ranges");
    group.sample_size(30);
    group.warm_up_time(Duration::from_secs(1));
    group.measurement_time(Duration::from_secs(3));
    for pages in [1, 10, 100, 1_000] {
        let (data, metadata, selection) = dataset(pages);
        group.bench_function(BenchmarkId::from_parameter(pages * 8), |b| {
            b.iter(|| build_reader(&data, &metadata, &selection));
        });
    }
    group.finish();
}

criterion_group!(benches, bench_page_ranges);
criterion_main!(benches);
