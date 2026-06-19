//! # fractus-core
//!
//! Cœur mathématique pur de fractus. Aucune I/O, aucune dépendance Python.
//! Toutes les fonctions ici sont testables en Rust seul.

pub mod vortex;

/// Addition entière. Existe uniquement pour le test fume Python↔Rust.
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
