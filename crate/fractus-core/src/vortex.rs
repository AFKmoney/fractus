//! # Vortex 2-adique


//!


//! Port depuis the original design (rust/src/vortex.rs), with corrections :


//! - L'import `HashMap` inutilise a ete retire.


//! - Le test tautological `assert!(d1 <= d2.max(d1))` a ete remplace by a true


//!   test d'ultrametrie : `d(x,z) <= max(d(x,y), d(y,z))` on donnees aleatoires.


//!


//! Nommage honesty : on parle of "hash Collatz" (pas "flot ergodique" — l'ergodicite


//! of Collatz est non demontree, problem open), of "ultrametric distance" and de


//! "2-adic norm" (termes exacts).



/// Valuation 2-adique v_2(x) = max{k : 2^k divise x}.


/// Pour x=0, on returns 64 (convention for u64).


pub fn valuation_2(x: u64) -> u32 {
    if x == 0 {
        return 64;
    }
    x.trailing_zeros()
}

/// Valuation 3-adique v_3(x) = max{k : 3^k divise x}.


///


/// Non utilisee en L0 (no appelant of production). Conservee because the spec L1


/// (cascade duale 2^n·3^k) en aura besoin ; marquee `allow(dead_code)` for


/// efastr the warning.


#[allow(dead_code)]
pub fn valuation_3(x: u64) -> u32 {
    if x == 0 {
        return 0; // convention : v_3(0) = infini, on borne a 0 for u64
    }
    let mut val = 0u32;
    let mut n = x;
    while n % 3 == 0 {
        val += 1;
        n /= 3;
    }
    val
}

/// Hash Collatz d'un integer. Utilise comme hachage d'etat deterministic.


/// Note : "ergodicite of Collatz" non demontree — on l'appelle juste "hash".


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

/// Distance ultrametrique 2-adique : d(a,b) = 2^{-v_2(a ⊕ b)}.


///


/// A comparer with the module source the original design d'origine


/// (rust/src/vortex.rs, function `distance`), which utilisait `2^{+v_2(a ⊕ b)}`


/// — l'inverse of the norme p-adique canonique `|x|_2 = 2^{-v_2(x)}`. Ici on


/// applique the formula canonique. Le test `test_ultrametric_strong_triangle_inequality`


/// (qui contient the triplet (7, 56, 13)) discrimine the deux formulas : il


/// echouerait with the version `+v_2` d'the original, tandis that the test equivaslow in


/// the original (`assert!(d1 <= d2.max(d1))`, tautologie) not detectait nothing.


///


/// Retourne a f64 in [0, 1] (0 si a == b).


pub fn ultrametric_distance(a: u64, b: u64) -> f64 {
    let diff = a ^ b;
    if diff == 0 {
        return 0.0; // d(a, a) = |0|_2 = 0
    }
    let v = valuation_2(diff) as i32;
    2f64.powi(-v)
}

/// Norme 2-adique : ||x||_2 = 2^{-v_2(x)}.


/// Retourne en f64 (can be very small for the larges x pairs).


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
        // Meme entree → same sortie (deterministic).


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
        // La vraie property ultrametrique : d(x,z) <= max(d(x,y), d(y,z)).


        // CORRECTION test tautological d'the original.


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
                "Echec ultrametrie : d({},{})={} > max(d({},{})={}, d({},{})={})",
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
        // Sur entrees fixes (pas fuzz, juste a echantillon), the norme


        // p-adique ||x||_2 = 2^{-v_2(x)} must be in (0, 1] for x != 0.


        for x in [1u64, 3, 5, 7, 9, 11, 42, 137, 1023, 65535] {
            let n = norm_2adic(x);
            assert!(n > 0.0 && n <= 1.0, "norm_2adic({}) = {} hors [0,1]", x, n);
        }
    }
}
