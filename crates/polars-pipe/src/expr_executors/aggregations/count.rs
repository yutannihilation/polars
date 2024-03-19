use super::*;

pub struct CountAgg {
    n: u64,
    include_nulls: bool
}

impl Aggregation for CountAgg {
    fn init(&mut self) {
        self.n = 0;
    }

    fn update(&mut self, batch: &Series) {
        if self.include_nulls {
            self.n += batch.len() as u64
        } else {
            self.n += (batch.len() - batch.null_count()) as u64;
        }
    }

    fn finalize(&mut self) -> AnyValue<'static> {
        AnyValue::UInt64(self.n)
    }

    fn combine(&mut self, other: &dyn Aggregation) {
        let other = other.as_any().downcast_ref::<CountAgg>().unwrap();
        self.n += other.n;
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}