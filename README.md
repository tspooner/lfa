# LFA ([api](https://docs.rs/lfa))

[![Crates.io](https://img.shields.io/crates/v/lfa.svg)](https://crates.io/crates/lfa)
[![Build Status](https://travis-ci.org/tspooner/lfa.svg?branch=master)](https://travis-ci.org/tspooner/lfa)
[![Coverage Status](https://coveralls.io/repos/github/tspooner/lfa/badge.svg?branch=master)](https://coveralls.io/github/tspooner/lfa?branch=master)


## Overview
`lfa` provides a set of implementations for common linear function
approximation techniques used in reinforcement learning.


## Installation
```toml
[dependencies]
lfa = "0.15"
```

Note that `rsrl` enables the `blas` feature of its [`ndarray`] dependency, so
if you're building a binary, you additionally need to specify a BLAS backend
compatible with `ndarray`. For example, you can add these dependencies:

[`ndarray`]: https://crates.io/crates/ndarray

```toml
blas-src = { version = "0.2.0", default-features = false, features = ["openblas"] }
openblas-src = { version = "0.6.0", default-features = false, features = ["cblas", "system"] }
```

See `ndarray`'s [README](https://github.com/rust-ndarray/ndarray#how-to-use-with-cargo)
for more information.

## Contributing
Pull requests are welcome. For major changes, please open an issue first to
discuss what you would like to change.

Please make sure to update tests as appropriate and adhere to the angularjs commit message conventions (see [here](https://gist.github.com/stephenparish/9941e89d80e2bc58a153)).

## License
[MIT](https://choosealicense.com/licenses/mit/)
