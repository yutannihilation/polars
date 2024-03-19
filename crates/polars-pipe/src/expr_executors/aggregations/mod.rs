use std::any::Any;
use polars_core::prelude::AnyValue;
use polars_core::series::Series;

mod count;
mod sum;

pub trait Aggregation {
    fn init(&mut self);

    fn update(&mut self, batch: &Series);

    fn finalize(&mut self) -> AnyValue<'static>;

    fn combine(&mut self, other: &dyn Aggregation);

    fn as_any(&self) -> &dyn Any;
}
