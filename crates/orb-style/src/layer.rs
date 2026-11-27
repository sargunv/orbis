//! A layer: its source and selection, and its properties in the order of
//! the property table ([`PropertyId`]).

use orb_expr::{Filter, PropertyExpression};

use crate::property::{PropertyRef, PropertyValue, Slot, Visibility};
use crate::spec::{Key, LayerType, PropertyId, Ramp};

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
    /// Paint properties' own `*-transition`, one per property that eases;
    /// the rest take the style's. Read through [`Layer::transition`].
    pub transitions: Vec<(PropertyId, crate::TransitionOverride)>,
    /// Layers with the same group select the same features and lay them
    /// out alike: their type, source, source layer, zoom range, filter and
    /// layout are the same, as GL JS's and Native's `groupByLayout` key
    /// them. `None` groups with no other layer.
    pub layout_group: Option<u64>,
    /// The render target the layer draws into rather than the map
    /// (`dev.sargunv.orbis:destination-target`).
    pub destination_target: Option<String>,
    /// The render target a fill draws through or an image-process layer
    /// processes; `Some` for every image-process layer.
    pub reads_target: Option<crate::TargetRead>,
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
            layout_group: None,
            destination_target: None,
            reads_target: None,
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

    /// The transition a paint change of property `id` eases by: its own
    /// `*-transition` over the style's `transition`.
    pub fn transition(&self, id: PropertyId, style: crate::Transition) -> crate::Transition {
        self.transitions
            .iter()
            .find(|(other, _)| *other == id)
            .map_or(style, |(_, t)| t.over(style))
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
