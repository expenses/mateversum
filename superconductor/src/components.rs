use bevy_ecs::prelude::{Component, Entity};
use renderer_core::assets::models;
use renderer_core::utils::Setter;
use std::ops::Range;

#[derive(Component)]
pub struct Instance(pub renderer_core::Instance);

#[derive(Component)]
pub struct InstanceOf(pub Entity);

#[derive(Component)]
pub struct PendingModel(pub Setter<models::Model>);

#[derive(Component)]
pub struct Model(pub models::Model);

#[derive(Component)]
pub struct Instances(pub Vec<renderer_core::Instance>);

#[derive(Component)]
pub struct InstanceRange(pub Range<u32>);

#[derive(Component)]
pub struct ModelUrl(pub url::Url);
