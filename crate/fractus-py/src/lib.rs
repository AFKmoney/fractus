//! Bindings Python (PyO3) pour fractus-core.
//!
//! Ce crate ne contient AUCUNE logique — seulement des wrappers #[pyfunction]
//! qui délèguent à fractus-core. Le but est d'exposer le Rust à Python
//! sous le nom `fractus._core`.

use pyo3::prelude::*;

/// Addition entière — wrapper Python pour fractus_core::add.
/// Exposée uniquement pour le test fume.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    fractus_core::add(a, b)
}

/// Hash Collatz d'un token id. Wrapper pour fractus_core::vortex::collatz_hash.
/// Utilisé comme conditionnement déterministe (hors-graphe autodiff) pour
/// l'embedding fractal (option B du spec L1).
#[pyfunction]
fn collatz_hash(x: u64, steps: u32) -> u64 {
    fractus_core::vortex::collatz_hash(x, steps)
}

/// Distance ultramétrique 2-adique : d(a,b) = 2^{-v_2(a ⊕ b)}.
/// Wrapper pour fractus_core::vortex::ultrametric_distance. Dans (0, 1].
#[pyfunction]
fn ultrametric_distance(a: u64, b: u64) -> f64 {
    fractus_core::vortex::ultrametric_distance(a, b)
}

/// Norme 2-adique : ||x||_2 = 2^{-v_2(x)}. Wrapper pour fractus_core::vortex::norm_2adic.
#[pyfunction]
fn norm_2adic(x: u64) -> f64 {
    fractus_core::vortex::norm_2adic(x)
}

/// Module Python `fractus._core`.
///
/// Signature pyo3 0.29 : le module est reçu comme `&Bound<'_, PyModule>`.
/// Les méthodes `.add_function(...)` viennent du trait `PyModuleMethods`
/// (ré-exporté par `pyo3::prelude`).
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(collatz_hash, m)?)?;
    m.add_function(wrap_pyfunction!(ultrametric_distance, m)?)?;
    m.add_function(wrap_pyfunction!(norm_2adic, m)?)?;
    Ok(())
}
