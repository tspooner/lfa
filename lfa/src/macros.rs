#![macro_use]

macro_rules! import_all {
    ($module:ident) => {
        mod $module;
        pub use self::$module::*;
    };
    ($module:ident with macros) => {
        #[macro_use]
        mod $module;
        pub use self::$module::*;
    };
}
