#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(target_arch = "spirv"))]
use crevice::std140::AsStd140;
use glam::{Mat4, Vec3, Vec4};

#[cfg_attr(not(target_arch = "spirv"), derive(crevice::std140::AsStd140))]
#[repr(C)]
pub struct Uniforms {
    pub projection_view: FlatMat4,
    pub eye_position: Vec3,
}

#[derive(Clone, Copy)]
#[cfg_attr(not(target_arch = "spirv"), derive(AsStd140))]
pub struct FlatMat4 {
    col_0: Vec4,
    col_1: Vec4,
    col_2: Vec4,
    col_3: Vec4,
}

impl From<FlatMat4> for Mat4 {
    fn from(d: FlatMat4) -> Self {
        Self::from_cols(d.col_0, d.col_1, d.col_2, d.col_3)
    }
}

impl From<Mat4> for FlatMat4 {
    fn from(mat: Mat4) -> Self {
        Self {
            col_0: mat.col(0),
            col_1: mat.col(1),
            col_2: mat.col(2),
            col_3: mat.col(3),
        }
    }
}

#[cfg_attr(not(target_arch = "spirv"), derive(AsStd140))]
pub struct MaterialSettings {
    pub base_color_factor: Vec4,
    pub emissive_factor: Vec3,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
}

// https://docs.gl/sl4/reflect
pub fn reflect(incident: Vec3, normal: Vec3) -> Vec3 {
    incident - 2.0 * normal.dot(incident) * normal
}

pub fn reflect_in_mirror(position: Vec3, mirror_position: Vec3, mirror_normal: Vec3) -> Vec3 {
    reflect(position - mirror_position, mirror_normal) + mirror_position
}

#[test]
fn mirror_logic_check() {
    let mirror_position = Vec3::new(0.0, -1.0, 0.0);
    let mirror_normal = Vec3::new(0.0, -1.0, 0.0);
    let tree_position = Vec3::new(0.0, 1.0, 0.0);

    dbg!(reflect_in_mirror(
        tree_position,
        mirror_position,
        mirror_normal
    ));

    panic!();
}

#[cfg_attr(not(target_arch = "spirv"), derive(AsStd140))]
pub struct MirrorUniforms {
    pub position: Vec3,
    pub normal: Vec3,
}
