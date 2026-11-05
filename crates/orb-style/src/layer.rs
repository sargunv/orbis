//! A layer: its source and selection, and its properties in the order of
//! the property table ([`PropertyId`]).

use orb_expr::{Filter, PropertyExpression};

use crate::property::{PropertyRef, PropertyValue, Slot, Visibility};
use crate::spec::{Key, LayerType, List, PropertyId, Ramp};

/// A style layer.
#[derive(Clone, Debug)]
pub struct Layer {
    pub id: String,
    pub ty: LayerType,
    /// The source drawn; `None` for a background.
    pub source: Option<String>,
    /// The vector source layer; `None` for GeoJSON and image sources.
    pub source_layer: Option<String>,
    /// Which features a feature layer draws.
    pub filter: Option<Filter>,
    /// The layer draws from this zoom, inclusive.
    pub minzoom: f64,
    /// The layer draws up to this zoom, exclusive. GL JS ignores a
    /// `maxzoom` of 0, which reads as unset.
    pub maxzoom: f64,
    pub visibility: Visibility,
    /// The layer type's properties, in table order.
    props: Box<[Slot]>,
    /// The table index of the type's first property.
    first: u16,
    /// Paint properties' own `*-transition`, by property name; the rest
    /// take the style's.
    pub transitions: Vec<(String, crate::TransitionOverride)>,
}
impl Layer {
    /// A layer of type `ty` with `props`, one slot per property of the
    /// type in table order, and the rest unset: for front ends to fill in.
    pub fn new(id: String, ty: LayerType, props: Vec<Slot>) -> Self {
        assert_eq!(
            props.len(),
            ty.properties().count(),
            "one slot per property"
        );
        Self {
            id,
            ty,
            source: None,
            source_layer: None,
            filter: None,
            minzoom: 0.0,
            maxzoom: f64::INFINITY,
            visibility: Visibility::Visible,
            props: props.into(),
            first: ty.first(),
            transitions: Vec::new(),
        }
    }

    /// Whether the layer draws at `zoom`, as GL JS's `isHidden` decides: a
    /// zero `minzoom` or `maxzoom` sets no bound, so default layers draw at
    /// the negative zooms a map can reach.
    pub fn is_visible_at(&self, zoom: f64) -> bool {
        let above_min = self.minzoom == 0.0 || zoom >= self.minzoom;
        let below_max = self.maxzoom == 0.0 || zoom < self.maxzoom;
        self.visibility == Visibility::Visible && above_min && below_max
    }

    #[inline]
    fn at(&self, id: PropertyId) -> usize {
        debug_assert_eq!(id.layer_type(), self.ty, "{} is another type's", id.name());
        usize::from(id.index() - self.first)
    }

    /// A property, read as its type.
    #[inline]
    pub fn get<T: PropertyValue>(&self, key: Key<T>) -> PropertyRef<'_, T> {
        PropertyRef::of(self.slot(key.id()), key.id())
    }

    /// A per-light property's values.
    pub fn list<T: PropertyValue>(&self, key: Key<List<T>>) -> Vec<PropertyRef<'_, T>> {
        self.slot(key.id())
            .items()
            .iter()
            .map(|slot| PropertyRef::of(slot, key.id()))
            .collect()
    }

    /// A colour ramp, if the layer has one.
    pub fn ramp(&self, key: Key<Ramp>) -> Option<&PropertyExpression> {
        self.slot(key.id()).as_expression()
    }

    /// A property, untyped.
    #[inline]
    pub fn slot(&self, id: PropertyId) -> &Slot {
        &self.props[self.at(id)]
    }

    pub fn slot_mut(&mut self, id: PropertyId) -> &mut Slot {
        let at = self.at(id);
        &mut self.props[at]
    }

    /// The layer's properties and their slots, in table order.
    pub fn slots(&self) -> impl Iterator<Item = (PropertyId, &Slot)> {
        self.ty.properties().zip(self.props.iter())
    }
}
