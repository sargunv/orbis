//! Decimal numbers to the nearest `f64`, ties to even, as `str::parse`
//! rounds them and `JSON.parse` does, without std's 10 KiB table of powers
//! of five, which the web's binary would carry and brotli can't shrink.
//!
//! A number of up to 19 significant digits that `f64` holds exactly, scaled
//! by a power of ten it holds exactly too, is one correctly rounded
//! operation (Clinger's fast path); up to 19 digits scaled by at most
//! 10^19 either way is exact in `u128`, as a product or a quotient and its
//! remainder; longer or more extreme numbers divide out exactly in big
//! integers, slowly.

/// `text` as the nearest `f64`, for decimal literals: an optional sign,
/// digits with an optional `.`, at least one digit, and an optional
/// exponent (`e` or `E`, an optional sign, digits). `None` for anything
/// else, where `str::parse` also fails, or reads `inf` or `nan`.
pub fn parse(text: &str) -> Option<f64> {
    let bytes = text.as_bytes();
    let (negative, mut i) = match bytes.first() {
        Some(b'-') => (true, 1),
        Some(b'+') => (false, 1),
        _ => (false, 0),
    };
    let digits_from = i;
    // The number is `w × 10^exponent`, `w` its first 19 significant digits
    // of `count`, without trailing zeros (`zeros` of them so far).
    let (mut w, mut count, mut zeros, mut exponent) = (0u64, 0usize, 0u32, 0i64);
    let mut any = false;
    let mut fraction = false;
    while let Some(&c) = bytes.get(i) {
        match c {
            b'0'..=b'9' => {
                any = true;
                if fraction {
                    exponent -= 1;
                }
                if c == b'0' {
                    zeros += u32::from(count > 0);
                } else {
                    count += zeros as usize + 1;
                    if count <= 19 {
                        w = w * 10u64.pow(zeros + 1) + u64::from(c - b'0');
                    }
                    zeros = 0;
                }
            }
            b'.' if !fraction => fraction = true,
            _ => break,
        }
        i += 1;
    }
    if !any {
        return None;
    }
    let digits_to = i;
    exponent += i64::from(zeros);
    if let Some(b'e' | b'E') = bytes.get(i) {
        i += 1;
        let sign = match bytes.get(i) {
            Some(b'-') => {
                i += 1;
                -1
            }
            Some(b'+') => {
                i += 1;
                1
            }
            _ => 1,
        };
        let start = i;
        let mut value: i64 = 0;
        while let Some(&c @ b'0'..=b'9') = bytes.get(i) {
            // Past this, the number is infinite or zero anyway.
            value = (value * 10 + i64::from(c - b'0')).min(1 << 40);
            i += 1;
        }
        if i == start {
            return None;
        }
        exponent += sign * value;
    }
    if i != bytes.len() {
        return None;
    }
    let magnitude = if count <= 19 {
        small(w, count, exponent)
    } else {
        // More digits than a `u64` holds: all of them, without leading or
        // trailing zeros.
        let digits: Vec<u8> = (bytes[digits_from..digits_to].iter())
            .filter(|c| c.is_ascii_digit())
            .map(|c| c - b'0')
            .skip_while(|&d| d == 0)
            .take(count)
            .collect();
        large(&digits, exponent)
    };
    Some(if negative { -magnitude } else { magnitude })
}

/// `w × 10^exponent`, `w` of `count` digits, 19 at most, with no trailing
/// zero.
fn small(w: u64, count: usize, exponent: i64) -> f64 {
    if w == 0 {
        return 0.0;
    }
    if let Some(out) = out_of_range(count, exponent) {
        return out;
    }
    let k = exponent.unsigned_abs() as usize;
    if w <= 1 << 53 && k <= 22 {
        let (w, power) = (w as f64, POWERS[k]);
        return if exponent < 0 { w / power } else { w * power };
    }
    if exponent >= 0 && k <= 19 {
        // Exact, and `as` rounds to nearest, ties to even.
        return (u128::from(w) * 10u128.pow(k as u32)) as f64;
    }
    if exponent < 0 && k <= 18 {
        let den = 10u128.pow(k as u32);
        // A quotient of 64 or 65 bits: `w << shift` fits, since `den` <
        // 2^60.
        let shift = 64 - (i64::from(bits(w.into())) - i64::from(bits(den)));
        let num = u128::from(w) << shift;
        return rounded(num / den, !num.is_multiple_of(den), shift);
    }
    let digits: Vec<u8> = w.to_string().bytes().map(|c| c - b'0').collect();
    large(&digits, exponent)
}

/// Infinity or zero for a number of `count` digits times 10^`exponent`
/// that is at least 10^309, past `f64::MAX`, or under 10^-324, under half
/// the least subnormal.
fn out_of_range(count: usize, exponent: i64) -> Option<f64> {
    let count = count as i64;
    if exponent + count - 1 > 308 {
        Some(f64::INFINITY)
    } else if exponent + count < -324 {
        Some(0.0)
    } else {
        None
    }
}

/// `digits × 10^exponent`, its digits without leading or trailing zeros,
/// divided out in big integers.
fn large(digits: &[u8], exponent: i64) -> f64 {
    if let Some(out) = out_of_range(digits.len(), exponent) {
        return out;
    }
    let mut num = Big::from_digits(digits);
    let mut den = Big::one();
    if exponent >= 0 {
        num.mul_pow10(exponent as u32);
    } else {
        den.mul_pow10(exponent.unsigned_abs() as u32);
    }
    // Scaled to a quotient of 64 or 65 bits.
    let shift = 64 - (i64::from(num.bits()) - i64::from(den.bits()));
    if shift > 0 {
        num.shl(shift as usize);
    } else {
        den.shl(shift.unsigned_abs() as usize);
    }
    let mut quotient = 0u128;
    for bit in (0..=65).rev() {
        let mut d = den.clone();
        d.shl(bit);
        if num.cmp(&d).is_ge() {
            num.sub(&d);
            quotient |= 1 << bit;
        }
    }
    rounded(quotient, !num.is_zero(), shift)
}

/// The nearest `f64` to `(quotient + ε) × 2^-shift`, where `quotient`
/// has 64 or 65 bits and `ε` is in (0, 1) when `inexact`, else 0.
fn rounded(mut quotient: u128, mut inexact: bool, mut shift: i64) -> f64 {
    if quotient >> 64 != 0 {
        inexact |= quotient & 1 != 0;
        quotient >>= 1;
        shift -= 1;
    }
    // The leading bit's power of two.
    let top = 63 - shift;
    // The bits below the mantissa's: those past 53, or more for a
    // subnormal, whose least bit is 2^-1074.
    let drop = if top >= -1022 { 11 } else { -1011 - top };
    if drop > 64 {
        return 0.0;
    }
    let drop = drop as u32;
    let mut mantissa = (quotient >> drop) as u64;
    let rest = quotient & ((1 << drop) - 1);
    let half = 1u128 << (drop - 1);
    if rest > half || (rest == half && (inexact || mantissa & 1 == 1)) {
        mantissa += 1;
    }
    if top < -1022 {
        // A subnormal, or the least normal when it rounded up into it.
        return f64::from_bits(mantissa);
    }
    let mut top = top;
    if mantissa == 1 << 53 {
        mantissa >>= 1;
        top += 1;
    }
    if top > 1023 {
        return f64::INFINITY;
    }
    f64::from_bits(((top + 1023) as u64) << 52 | (mantissa & ((1 << 52) - 1)))
}

/// 10^0 to 10^22, each exact in `f64`.
const POWERS: [f64; 23] = {
    let mut powers = [1.0; 23];
    let mut i = 1;
    while i < 23 {
        powers[i] = powers[i - 1] * 10.0;
        i += 1;
    }
    powers
};

/// The bits `x` needs.
fn bits(x: u128) -> u32 {
    128 - x.leading_zeros()
}

/// A natural number in 32-bit limbs, least first.
#[derive(Clone)]
struct Big(Vec<u32>);

impl Big {
    fn one() -> Self {
        Self(vec![1])
    }

    fn from_digits(digits: &[u8]) -> Self {
        let mut big = Self(Vec::new());
        for chunk in digits.chunks(9) {
            let value = chunk.iter().fold(0u32, |v, &d| v * 10 + u32::from(d));
            big.mul_add(10u32.pow(chunk.len() as u32), value);
        }
        big
    }

    /// `self × factor + add`.
    fn mul_add(&mut self, factor: u32, add: u32) {
        let mut carry = u64::from(add);
        for limb in &mut self.0 {
            let x = u64::from(*limb) * u64::from(factor) + carry;
            *limb = x as u32;
            carry = x >> 32;
        }
        if carry != 0 {
            self.0.push(carry as u32);
        }
    }

    fn mul_pow10(&mut self, mut n: u32) {
        while n > 0 {
            let k = n.min(9);
            self.mul_add(10u32.pow(k), 0);
            n -= k;
        }
    }

    fn shl(&mut self, n: usize) {
        let (limbs, bits) = (n / 32, n % 32);
        if bits != 0 {
            let mut carry = 0;
            for limb in &mut self.0 {
                let next = *limb >> (32 - bits);
                *limb = *limb << bits | carry;
                carry = next;
            }
            if carry != 0 {
                self.0.push(carry);
            }
        }
        self.0.splice(0..0, std::iter::repeat_n(0, limbs));
    }

    fn bits(&self) -> u32 {
        let top = self.0.iter().rposition(|&l| l != 0).map_or(0, |i| i + 1);
        match top {
            0 => 0,
            n => 32 * n as u32 - self.0[n - 1].leading_zeros(),
        }
    }

    fn is_zero(&self) -> bool {
        self.0.iter().all(|&l| l == 0)
    }

    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        let limb = |x: &Self, i: usize| x.0.get(i).copied().unwrap_or(0);
        (0..self.0.len().max(other.0.len()))
            .rev()
            .map(|i| limb(self, i).cmp(&limb(other, i)))
            .find(|o| o.is_ne())
            .unwrap_or(std::cmp::Ordering::Equal)
    }

    /// `self - other`, which isn't negative.
    fn sub(&mut self, other: &Self) {
        let mut borrow = 0i64;
        for (i, limb) in self.0.iter_mut().enumerate() {
            let x = i64::from(*limb) - i64::from(other.0.get(i).copied().unwrap_or(0)) - borrow;
            borrow = i64::from(x < 0);
            *limb = (x + (borrow << 32)) as u32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::parse;

    /// `parse` agrees with std's parser, bit for bit.
    fn agrees(text: &str) {
        let std: Result<f64, _> = text.parse();
        let ours = parse(text);
        match std {
            Ok(v) => assert_eq!(
                ours.map(f64::to_bits),
                Some(v.to_bits()),
                "{text}: {ours:?}, not {v:?}"
            ),
            Err(_) => assert_eq!(ours, None, "{text}"),
        }
    }

    #[test]
    fn rounds_as_std_does() {
        for text in [
            "0",
            "-0",
            "+0",
            "0.0",
            "00.000",
            "1",
            "-1",
            "0.1",
            "0.3",
            "1.5",
            ".5",
            "5.",
            "-.5",
            "1e0",
            "1E5",
            "1e-5",
            "1e+5",
            "123.456e7",
            "0.000001",
            "9007199254740992",
            "9007199254740993",
            "9007199254740995",
            "18446744073709551615",
            "18446744073709551616",
            "1e22",
            "1e23",
            "123456789012345678901234567890",
            "0.30000000000000004",
            "-122.41941550000001",
            "37.774929500000006",
            "2.2250738585072011e-308",
            "2.2250738585072014e-308",
            "4.9e-324",
            "5e-324",
            "2.4703282292062327e-324",
            "2.4703282292062328e-324",
            "1e-324",
            "1.7976931348623157e308",
            "1.7976931348623158e308",
            "1.7976931348623159e308",
            "1e308",
            "1e309",
            "1e-400",
            "1e400",
            "8.98846567431158e307",
            "4.940656458412465441765687928682213723651e-324",
            "7.2057594037927933e16",
            "1448997445238699",
            "1e100000000000",
            "1e-100000000000",
            "0.000000000000000000000000000000000000000000000000000000000000000000001",
            "0001.2300",
            "1000000000000000000000000",
            "10000000000000000000000000000000000000001",
            "1234567890123456789.0000000000001",
            "123456789012345678900000",
            "0.00000000000000000000123456789012345678901",
            "9999999999999999999e-30",
            "1.0e-300",
        ] {
            agrees(text);
        }
        // A very long number, and one at the halfway point between two
        // `f64`s that only its last digit decides.
        agrees(&format!("{}e-400", "1".repeat(500)));
        agrees("9007199254740993.0000000000000000000000000000001");
    }

    #[test]
    fn refuses_what_is_not_a_decimal_literal() {
        for text in [
            "", "-", "+", ".", "e5", "1e", "1e+", "1.2.3", "1x", " 1", "inf", "nan", "--1", "0x10",
        ] {
            assert_eq!(parse(text), None, "{text}");
        }
    }

    #[test]
    fn agrees_on_many_numbers() {
        // A fixed sequence of numbers of 1 to 24 digits, scaled across
        // `f64`'s whole range.
        let mut state = 0x9e37_79b9_7f4a_7c15u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..50_000 {
            let count = 1 + next() % 24;
            let digits: String = (0..count)
                .map(|_| char::from(b'0' + (next() % 10) as u8))
                .collect();
            let exponent = (next() % 700) as i64 - 350;
            let point = (next() % (count + 1)) as usize;
            let text = format!("{}.{}e{exponent}", &digits[..point], &digits[point..]);
            agrees(&text);
            agrees(&digits);
        }
    }
}
