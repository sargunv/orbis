//! The style's terrain from one turn to the next: on, off, and the
//! frames between, which wait for the pipelines the map draws with once
//! the turn is done, and ease the terrain's presence in or out.
//!
//! [`Map::step_terrain`] is the one place the phase moves, on what the
//! style, its DEM source and the renderer's pipelines say, never on the
//! camera's path: everything else reads it. Its code stays out of the
//! DEM module (`crate::modules`), which a map without terrain never
//! loads: what it calls of a terrain runs only once it holds one.

use std::sync::Arc;

use orb_hal::Gpu;
use orb_render::Renderer;
use orb_source::endpoint::TileEndpoint;

use crate::map::Map;
use crate::resident::SourceState;
use crate::root::RootFrame;
use crate::terrain::Terrain;

/// Where the style's terrain is in its turns on and off.
#[derive(Debug, Default)]
pub(crate) enum TerrainPhase {
    /// No terrain, nor any to come.
    #[default]
    Off,
    /// The style's terrain is to come: its DEM source resolves, or the
    /// map, having drawn flat, waits for the pipelines it draws with on
    /// terrain (`waits`), holding the terrain's rise until it can draw it
    /// ([`Renderer::set_terrain_reachable`]).
    Arming { waits: bool },
    /// The terrain stands, risen or rising.
    On(Terrain),
    /// The style dropped its terrain, which eases flat with its DEM and
    /// drape kept, so every frame of the fall draws the whole map.
    Falling(Terrain),
    /// Fallen flat, or still standing where the style names a terrain
    /// it can no longer draw (its DEM source failed, or isn't the
    /// surface's, or the device failed to make its pipelines), the
    /// terrain waits for the pipelines the map draws with
    /// without it ([`Renderer::terrain_ready_as`]) before it goes.
    Disarming(Terrain),
}

impl TerrainPhase {
    /// The terrain the map holds, standing, falling or fallen.
    pub(crate) fn get(&self) -> Option<&Terrain> {
        match self {
            Self::On(t) | Self::Falling(t) | Self::Disarming(t) => Some(t),
            Self::Off | Self::Arming { .. } => None,
        }
    }

    pub(crate) fn get_mut(&mut self) -> Option<&mut Terrain> {
        match self {
            Self::On(t) | Self::Falling(t) | Self::Disarming(t) => Some(t),
            Self::Off | Self::Arming { .. } => None,
        }
    }

    /// Takes the terrain the map holds, leaving the phase off.
    pub(crate) fn take(&mut self) -> Option<Terrain> {
        match std::mem::take(self) {
            Self::On(t) | Self::Falling(t) | Self::Disarming(t) => Some(t),
            Self::Off | Self::Arming { .. } => None,
        }
    }

    /// Whether a turn waits for pipelines: the map isn't idle while it
    /// does, whatever else has settled.
    pub(crate) fn waits(&self) -> bool {
        matches!(self, Self::Arming { waits: true } | Self::Disarming(_))
    }
}

/// The terrain a style asks for this frame, as the map declared it to
/// the renderer ([`Map::declare_terrain`]).
pub(crate) struct Declared {
    /// Its DEM source's index, and the source's endpoint once it is
    /// ready.
    wanted: Option<(usize, Option<Arc<TileEndpoint>>)>,
    /// The renderer takes it: it has drawn nothing flat, or what it draws
    /// on terrain is made.
    reachable: bool,
}

impl Map {
    /// Tells the renderer whether the style has terrain, on or still to
    /// come, before the frame's pipeline set is renewed: the set then
    /// holds what the map draws on terrain. `surface_full`: the surface
    /// needs the full projection without terrain.
    pub(crate) fn declare_terrain(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        surface_full: bool,
    ) -> Declared {
        let wanted = self.wanted_terrain(renderer.terrain_failed(gpu));
        // Views need the full projection on terrain, which the style says
        // it has before its DEM source resolves.
        renderer.set_full_projection(surface_full || wanted.is_some());
        let reachable = renderer.set_terrain_reachable(gpu, wanted.is_some());
        Declared { wanted, reachable }
    }

    /// Moves the style's terrain phase for this frame, on what the map
    /// `declared`, and eases the root's values (`zoom` the camera's),
    /// which the phase both holds and reads: a terrain that waits for its
    /// pipelines rises once it draws, and a falling one goes once it lies
    /// flat. Returns the frame's root values.
    pub(crate) fn step_terrain(
        &mut self,
        gpu: &mut dyn Gpu,
        renderer: &mut Renderer,
        Declared { wanted, reachable }: Declared,
        zoom: f64,
    ) -> RootFrame {
        let waits = wanted.is_some() && !reachable;
        if waits {
            self.root.hold_terrain();
        }
        let model = (self.style.as_ref()).map(|s| Arc::clone(&s.worker.plan().style));
        let root = self.root.frame(model.as_deref(), zoom, self.frame_time);
        let Some(style) = self.style.as_mut() else {
            return root;
        };
        let plan = style.worker.plan();
        let held = style.terrain.take();
        style.terrain = match (held, wanted) {
            (None, None) => TerrainPhase::Off,
            // Turned off, it falls flat as `root` eases it, and goes once
            // the map's pipelines without it are made. A terrain the style
            // still has but can't draw (its DEM source failed, or isn't the
            // surface's, or its pipelines failed) would never ease flat: it
            // goes without falling.
            (Some(terrain), None) => {
                if root.terrain_presence > 0.0 && plan.terrain().is_none() {
                    TerrainPhase::Falling(terrain)
                } else if !renderer.terrain_ready_as(gpu, false) {
                    TerrainPhase::Disarming(terrain)
                } else {
                    terrain.release(gpu, renderer);
                    TerrainPhase::Off
                }
            }
            (held, Some((index, endpoint))) => {
                let key = plan.sources[index].key;
                match held {
                    // Its source stays: a falling terrain rises again.
                    Some(mut terrain) if terrain.key == key => {
                        terrain.source = index;
                        TerrainPhase::On(terrain)
                    }
                    held => {
                        if let Some(old) = held {
                            old.release(gpu, renderer);
                        }
                        match endpoint.filter(|_| reachable) {
                            Some(endpoint) => {
                                // Raster sources cover with `tileSize / √2`
                                // (GL JS's rounded raster cover).
                                let tile_px = endpoint.cover_px * std::f64::consts::SQRT_2;
                                self.terrains += 1;
                                TerrainPhase::On(Terrain::new(
                                    self.terrains,
                                    (index, key),
                                    root.exaggeration,
                                    endpoint,
                                    self.surface.tms(),
                                    tile_px,
                                ))
                            }
                            None => TerrainPhase::Arming { waits },
                        }
                    }
                }
            }
        };
        root
    }

    /// The terrain the style asks for, unless the renderer's device
    /// `failed` to make what it draws with, or its DEM source failed, or
    /// its tiles can't be the surface's (saying so): terrain tiles are the
    /// surface's, and their DEM must be too.
    fn wanted_terrain(&mut self, failed: bool) -> Option<(usize, Option<Arc<TileEndpoint>>)> {
        let style = self.style.as_ref()?;
        let (index, _) = style.worker.plan().terrain()?;
        if failed {
            // The driver's failure is named with the device's stats; a
            // terrain drawn without its pipelines would lose its drape.
            let warning = "terrain: the GPU failed to make its pipelines; the map is flat";
            self.warnings.push(warning.to_owned());
            return None;
        }
        match &style.sources[index].state {
            SourceState::Resolving => Some((index, None)),
            SourceState::Ready { endpoint, .. } if endpoint.tms == *self.surface.tms() => {
                Some((index, Some(Arc::clone(endpoint))))
            }
            SourceState::Ready { endpoint, .. } => {
                let warning = format!(
                    "terrain: the DEM source's tiles ({}) are not the map's ({}); the map is flat",
                    endpoint.tms.id(),
                    self.surface.tms().id()
                );
                self.warnings.push(warning);
                None
            }
            SourceState::Failed => None,
        }
    }
}
