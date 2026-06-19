//! Bindings Python (PyO3) pour fractus-core.
//!
//! Ce crate ne contient AUCUNE logique — seulement des wrappers #[pyfunction]
//! qui délèguent à fractus-core. Le but est d'exposer le Rust à Python
//! sous le nom `fractus._core`.

use pyo3::prelude::*;

/// Addition entière — wrapper Python pour fractus_core::add.
/// Exposée uniquement pour le test fume en L0.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    fractus_core::add(a, b)
}

/// Module Python `fractus._core`.
///
/// Signature pyo3 0.29 : le module est reçu comme `&Bound<'_, PyModule>`.
/// Les méthodes `.add_function(...)` viennent du trait `PyModuleMethods`
/// (ré-exporté par `pyo3::prelude`).
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    Ok(())
}
