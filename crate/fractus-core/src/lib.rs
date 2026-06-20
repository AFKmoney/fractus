//! # fractus-core

//!

//! Coeur mathematical pur of fractus. Aucune I/O, no dependance Python.

//! Toutes the functions ici are testables en Rust seul.


pub mod vortex;

/// Addition integere. Existe uniquement for the test fume Python↔Rust.

pub fn add(a: i64, b: i64) -> i64 {
    a + b
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_add() {
        assert_eq!(add(2, 3), 5);
        assert_eq!(add(-1, 1), 0);
    }
}
