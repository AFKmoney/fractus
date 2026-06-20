//! Bindings Python (PyO3) for fractus-core.
//!
//! Ce crate ne contient AUCUNE logical — seulement des wrappers #[pyfunction]
//! qui deleguent a fractus-core. Le but est d'exposer le Rust a Python
//! under le nom `fractus._core`.

use pyo3::prelude::*;

/// Addition entiere — wrapper Python for fractus_core::add.
/// Exposee uniquement for le test fume.
#[pyfunction]
fn add(a: i64, b: i64) -> i64 {
    fractus_core::add(a, b)
}

/// Hash Collatz d'un token id. Wrapper for fractus_core::vortex::collatz_hash.
/// Utilise comme conditionnement deterministe (hors-graphe autodiff) for
/// l'embedding fractal (option B du spec L1).
#[pyfunction]
fn collatz_hash(x: u64, steps: u32) -> u64 {
    fractus_core::vortex::collatz_hash(x, steps)
}

/// Distance ultrametrique 2-adique : d(a,b) = 2^{-v_2(a ⊕ b)}.
/// Wrapper for fractus_core::vortex::ultrametric_distance. Dans (0, 1].
#[pyfunction]
fn ultrametric_distance(a: u64, b: u64) -> f64 {
    fractus_core::vortex::ultrametric_distance(a, b)
}

/// Norme 2-adique : ||x||_2 = 2^{-v_2(x)}. Wrapper for fractus_core::vortex::norm_2adic.
#[pyfunction]
fn norm_2adic(x: u64) -> f64 {
    fractus_core::vortex::norm_2adic(x)
}

/// Module Python `fractus._core`.
///
/// Signature pyo3 0.29 : le module est recu comme `&Bound<'_, PyModule>`.
/// Les methodes `.add_function(...)` viennent du trait `PyModuleMethods`
/// (re-exporte par `pyo3::prelude`).
#[pymodule]
fn _core(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(add, m)?)?;
    m.add_function(wrap_pyfunction!(collatz_hash, m)?)?;
    m.add_function(wrap_pyfunction!(ultrametric_distance, m)?)?;
    m.add_function(wrap_pyfunction!(norm_2adic, m)?)?;
    Ok(())
}
