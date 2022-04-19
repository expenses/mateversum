#![no_std]

use glam::{Mat4, Vec3, Vec4};

#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct Uniforms {
    pub projection_view: DecompMat,
    pub eye_position: Vec3,
    #[cfg(not(target_arch = "spirv"))]
    pub _padding: u32,
}

#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct DecompMat {
    col_0: Vec4,
    col_1: Vec4,
    col_2: Vec4,
    col_3: Vec4,
}

impl From<DecompMat> for Mat4 {
    fn from(d: DecompMat) -> Self {
        Self::from_cols(d.col_0, d.col_1, d.col_2, d.col_3)
    }
}

impl From<Mat4> for DecompMat {
    fn from(mat: Mat4) -> Self {
        Self {
            col_0: mat.col(0),
            col_1: mat.col(1),
            col_2: mat.col(2),
            col_3: mat.col(3),
        }
    }
}

#[derive(Clone, Copy, Default)]
#[cfg_attr(not(target_arch = "spirv"), derive(bytemuck::Pod, bytemuck::Zeroable))]
#[repr(C)]
pub struct MaterialSettings {
    pub base_color_factor: Vec4,
    pub emissive_factor: Vec3,
    #[cfg(not(target_arch = "spirv"))]
    pub _padding0: u32,
    pub metallic_factor: f32,
    #[cfg(not(target_arch = "spirv"))]
    pub _padding1: u32,
    pub roughness_factor: f32,
    #[cfg(not(target_arch = "spirv"))]
    pub _padding2: u32,
}
