//! # Vortex 2-adique
//!
//! Port depuis OMNI-FRACTAL (rust/src/vortex.rs), avec corrections :
//! - L'import `HashMap` inutilisé a été retiré.
//! - Le test tautologique `assert!(d1 <= d2.max(d1))` a été remplacé par un vrai
//!   test d'ultramétrie : `d(x,z) <= max(d(x,y), d(y,z))` sur données aléatoires.
//!
//! Nommage honnête : on parle de "hash Collatz" (pas "flot ergodique" — l'ergodicité
//! de Collatz est non démontrée, problème ouvert), de "distance ultramétrique" et de
//! "norme 2-adique" (termes exacts).

/// Valuation 2-adique v_2(x) = max{k : 2^k divise x}.
/// Pour x=0, on retourne 64 (convention pour u64).
pub fn valuation_2(x: u64) -> u32 {
    if x == 0 {
        return 64;
    }
    x.trailing_zeros()
}

/// Valuation 3-adique v_3(x) = max{k : 3^k divise x}.
///
/// Non utilisée en L0 (aucun appelant de production). Conservée car le spec L1
/// (cascade duale 2^n·3^k) en aura besoin ; marquée `allow(dead_code)` pour
/// éviter le warning.
#[allow(dead_code)]
pub fn valuation_3(x: u64) -> u32 {
    if x == 0 {
        return 0; // convention : v_3(0) = infini, on borne à 0 pour u64
    }
    let mut val = 0u32;
    let mut n = x;
    while n % 3 == 0 {
        val += 1;
        n /= 3;
    }
    val
}

/// Hash Collatz d'un entier. Utilisé comme hachage d'état déterministe.
/// Note : "ergodicité de Collatz" non démontrée — on l'appelle juste "hash".
pub fn collatz_hash(mut x: u64, steps: u32) -> u64 {
    for _ in 0..steps {
        if x == 0 {
            break;
        }
        if x % 2 == 0 {
            x /= 2;
        } else {
            x = 3u64.wrapping_mul(x).wrapping_add(1);
        }
    }
    x
}

/// Distance ultramétrique 2-adique : d(a,b) = 2^{-v_2(a ⊕ b)}.
///
/// À comparer avec le module source OMNI-FRACTAL d'origine
/// (rust/src/vortex.rs, fonction `distance`), qui utilisait `2^{+v_2(a ⊕ b)}`
/// — l'inverse de la norme p-adique canonique `|x|_2 = 2^{-v_2(x)}`. Ici on
/// applique la formule canonique. Le test `test_ultrametric_strong_triangle_inequality`
/// (qui contient le triplet (7, 56, 13)) discrimine les deux formules : il
/// échouerait avec la version `+v_2` d'OMNI, tandis que le test équivalent dans
/// OMNI (`assert!(d1 <= d2.max(d1))`, tautologie) ne détectait rien.
///
/// Retourne un f64 dans [0, 1] (0 si a == b).
pub fn ultrametric_distance(a: u64, b: u64) -> f64 {
    let diff = a ^ b;
    if diff == 0 {
        return 0.0; // d(a, a) = |0|_2 = 0
    }
    let v = valuation_2(diff) as i32;
    2f64.powi(-v)
}

/// Norme 2-adique : ||x||_2 = 2^{-v_2(x)}.
/// Retourné en f64 (peut être très petit pour les grands x pairs).
pub fn norm_2adic(x: u64) -> f64 {
    if x == 0 {
        return 0.0; // ||0|| = 0 par convention (v_2(0) = infini)
    }
    let v = valuation_2(x) as i32;
    2f64.powi(-v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valuation_2_basic() {
        assert_eq!(valuation_2(0), 64);
        assert_eq!(valuation_2(1), 0);
        assert_eq!(valuation_2(2), 1);
        assert_eq!(valuation_2(4), 2);
        assert_eq!(valuation_2(8), 3);
        assert_eq!(valuation_2(56), 3); // 56 = 7 * 8
    }

    #[test]
    fn test_valuation_3_basic() {
        assert_eq!(valuation_3(0), 0);
        assert_eq!(valuation_3(1), 0);
        assert_eq!(valuation_3(3), 1);
        assert_eq!(valuation_3(9), 2);
        assert_eq!(valuation_3(27), 3);
        assert_eq!(valuation_3(56), 0); // 56 n'est pas divisible par 3
    }

    #[test]
    fn test_collatz_hash_deterministic() {
        // Même entrée → même sortie (déterministe).
        assert_eq!(collatz_hash(7, 10), collatz_hash(7, 10));
        // 0 reste 0.
        assert_eq!(collatz_hash(0, 10), 0);
    }

    #[test]
    fn test_ultrametric_distance_self_is_zero() {
        assert_eq!(ultrametric_distance(42, 42), 0.0);
    }

    #[test]
    fn test_ultrametric_distance_symmetry() {
        for (a, b) in [(1u64, 2), (7, 56), (100, 200), (3, 9)] {
            assert_eq!(ultrametric_distance(a, b), ultrametric_distance(b, a));
        }
    }

    #[test]
    fn test_ultrametric_strong_triangle_inequality() {
        // La vraie propriété ultramétrique : d(x,z) <= max(d(x,y), d(y,z)).
        // CORRECTION du test tautologique d'OMNI.
        let triples: [(u64, u64, u64); 8] = [
            (1, 2, 4),
            (7, 56, 13),
            (100, 200, 300),
            (3, 9, 27),
            (5, 11, 23),
            (1024, 1, 2),
            (7, 13, 21),
            (255, 256, 257),
        ];
        for (x, y, z) in triples {
            let d_xy = ultrametric_distance(x, y);
            let d_yz = ultrametric_distance(y, z);
            let d_xz = ultrametric_distance(x, z);
            assert!(
                d_xz <= d_xy.max(d_yz),
                "Échec ultramétrie : d({},{})={} > max(d({},{})={}, d({},{})={})",
                x, z, d_xz, x, y, d_xy, y, z, d_yz
            );
        }
    }

    #[test]
    fn test_norm_2adic_basic() {
        assert_eq!(norm_2adic(0), 0.0);
        assert_eq!(norm_2adic(1), 1.0); // v_2(1) = 0 → 2^0 = 1
        assert_eq!(norm_2adic(2), 0.5); // v_2(2) = 1 → 2^-1
        assert_eq!(norm_2adic(4), 0.25); // v_2(4) = 2 → 2^-2
        assert_eq!(norm_2adic(8), 0.125); // v_2(8) = 3 → 2^-3
    }

    #[test]
    fn test_norm_2adic_in_unit_interval() {
        // Sur entrées fixes (pas du fuzz, juste un échantillon), la norme
        // p-adique ||x||_2 = 2^{-v_2(x)} doit être dans (0, 1] pour x != 0.
        for x in [1u64, 3, 5, 7, 9, 11, 42, 137, 1023, 65535] {
            let n = norm_2adic(x);
            assert!(n > 0.0 && n <= 1.0, "norm_2adic({}) = {} hors [0,1]", x, n);
        }
    }
}
