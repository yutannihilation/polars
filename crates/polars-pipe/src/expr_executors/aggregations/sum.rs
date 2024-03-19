use super::*;
use polars_core::prelude::*;
use polars_core::datatypes::{NumericNative, PolarsNumericType};

pub struct Sum<T: NumericNative> {
    state: T
}

impl<T> Aggregation for Sum<T>
where T: NumericNative,
for<'a> &'a ChunkedArray<T::PolarsType>: ChunkAgg<T>

{
    fn init(&mut self) {
        self.state = T::zero();
    }

    fn update(&mut self, batch: &Series) {
        let phys = batch.to_physical_repr();
        let ca = phys.unpack::<T::PolarsType>().unwrap();
        self.state += ca.sum().unwrap();
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        AnyValue::from(self.state)
    }

    fn combine(&mut self, other: &dyn Aggregation) {
        let other = other.as_any().downcast_ref::<Sum<T>>().unwrap();
        self.state += other.state;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
