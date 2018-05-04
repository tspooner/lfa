/// ! Linear basis projection module.

mod rbf_network;
pub use self::rbf_network::*;

mod fourier;
pub use self::fourier::*;

mod polynomial;
pub use self::polynomial::*;

mod tile_coding;
pub use self::tile_coding::*;

mod uniform_grid;
pub use self::uniform_grid::*;

mod random;
pub use self::random::*;
