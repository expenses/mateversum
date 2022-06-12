#![cfg_attr(not(feature = "std"), no_std)]

#[cfg(not(target_arch = "spirv"))]
use crevice::std140::AsStd140;
use glam::{Mat4, Vec3, Vec4};

#[cfg_attr(not(target_arch = "spirv"), derive(crevice::std140::AsStd140))]
#[repr(C)]
pub struct Uniforms {
    pub left_projection_view: FlatMat4,
    pub right_projection_view: FlatMat4,
    pub left_eye_position: Vec3,
    pub right_eye_position: Vec3,
}

impl Uniforms {
    pub fn projection_view(&self, view_index: i32) -> Mat4 {
        Mat4::from(if view_index != 0 {
            self.right_projection_view
        } else {
            self.left_projection_view
        })
    }

    pub fn eye_position(&self, view_index: i32) -> Vec3 {
        if view_index != 0 {
            self.right_eye_position
        } else {
            self.left_eye_position
        }
    }
}

#[cfg_attr(not(target_arch = "spirv"), derive(crevice::std140::AsStd140))]
#[repr(C)]
pub struct SkyboxUniforms {
    pub left_projection_inverse: FlatMat4,
    pub right_projection_inverse: FlatMat4,
    pub left_view_inverse: Vec4,
    pub right_view_inverse: Vec4,
}

impl SkyboxUniforms {
    pub fn projection_inverse(&self, view_index: i32) -> Mat4 {
        Mat4::from(if view_index != 0 {
            self.right_projection_inverse
        } else {
            self.left_projection_inverse
        })
    }

    pub fn view_inverse(&self, view_index: i32) -> Vec4 {
        if view_index != 0 {
            self.right_view_inverse
        } else {
            self.left_view_inverse
        }
    }
}

#[derive(Clone, Copy, Default)]
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
    pub toon_shading: ToonShadingSettings,
    pub base_color_factor: Vec4,
    pub emissive_factor: Vec3,
    pub metallic_factor: f32,
    pub roughness_factor: f32,
    pub mode: u32,
}

#[cfg_attr(not(target_arch = "spirv"), derive(AsStd140, Debug))]
#[derive(Default)]
pub struct ToonShadingSettings {
    pub shade_colour_factor: Vec3,
    pub shift_factor: f32,
    pub toony_factor: f32,
}

pub mod mode {
    pub const PBR: u32 = 0;
    pub const UNLIT: u32 = 1;
    pub const TOON: u32 = 2;
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
