//! What a view draws on (plan §9.1).

use crate::Projection;

/// What a view draws on (plan §9.1): its projection's plane or, at low
/// zoom, a globe the plane's tiles wrap around. Style zoom is always the
/// plane's (on Web Mercator, GL JS's convention on its globe too).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Surface {
    pub projection: Projection,
    pub globe: Option<Globe>,
}

impl Surface {
    pub fn plane(projection: Projection) -> Self {
        Self {
            projection,
            globe: None,
        }
    }

    /// How much of a globe a view at style zoom `zoom` is: 1 a globe, 0
    /// the plane, and a blend of the two between (GPU positions mix; the
    /// CPU uses the globe).
    pub fn globe_amount(&self, zoom: f64) -> f64 {
        self.globe.map_or(0.0, |globe| globe.amount(zoom))
    }
}

impl From<Projection> for Surface {
    fn from(projection: Projection) -> Self {
        Self::plane(projection)
    }
}

impl From<&Projection> for Surface {
    fn from(projection: &Projection) -> Self {
        Self::plane(*projection)
    }
}

/// A sphere the plane's tiles wrap around ([`sphere`](crate::sphere)), handing over
/// to the plane between two zooms.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Globe {
    /// A globe through the first zoom, the plane from the second; `None`
    /// keeps the globe at every zoom (v8 `vertical-perspective`).
    pub handover: Option<[f64; 2]>,
}

impl Globe {
    /// v8 `globe`: a globe through zoom 11 and Web Mercator from zoom 12,
    /// as GL JS hands over (`projection_factory.ts`).
    pub const V8: Self = Self {
        handover: Some([11.0, 12.0]),
    };

    /// How much of a globe a view at `zoom` is.
    pub fn amount(&self, zoom: f64) -> f64 {
        match self.handover {
            Some([from, to]) => ((to - zoom) / (to - from)).clamp(0.0, 1.0),
            None => 1.0,
        }
    }
}
