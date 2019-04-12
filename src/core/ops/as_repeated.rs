pub trait AsRepeated<T> {
    fn as_repeated(self) -> T;
}

impl<T> AsRepeated<T> for T {
    fn as_repeated(self) -> T { self }
}

impl AsRepeated<[f64; 2]> for f64 {
    fn as_repeated(self) -> [f64; 2] { [self, self] }
}

impl AsRepeated<[f64; 3]> for f64 {
    fn as_repeated(self) -> [f64; 3] { [self, self, self] }
}
