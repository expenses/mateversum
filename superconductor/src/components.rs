use bevy_ecs::prelude::{Component, Entity};
use renderer_core::arc_swap;
use renderer_core::assets::models;
use std::ops::Range;
use std::sync::Arc;

#[derive(Component)]
pub struct Instance(pub renderer_core::Instance);

#[derive(Component)]
pub struct InstanceOf(pub Entity);

#[derive(Component)]
pub struct PendingModel(pub Arc<arc_swap::ArcSwapOption<models::Model>>);

#[derive(Component)]
pub struct Model(pub models::Model);

#[derive(Component)]
pub struct Instances(pub Vec<renderer_core::Instance>);

#[derive(Component)]
pub struct InstanceRange(pub Range<u32>);

#[derive(Component)]
pub struct ModelUrl(pub url::Url);
