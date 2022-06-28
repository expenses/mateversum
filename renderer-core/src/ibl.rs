use crate::Texture;
use arc_swap::ArcSwap;

pub struct IblTextures {
    pub ggx_lut: ArcSwap<Texture>,
    pub diffuse_cubemap: ArcSwap<Texture>,
    pub specular_cubemap: ArcSwap<Texture>,
}
