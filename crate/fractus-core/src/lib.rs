//! # fractus-core
//!
//! Coeur mathematical pur de fractus. Aucune I/O, no dependance Python.
//! Toutes les functions ici sont testables en Rust seul.

pub mod vortex;

/// Addition entiere. Existe uniquement for le test fume Python↔Rust.
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
