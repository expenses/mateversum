#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr, asm_const, asm_experimental_arch),
    register_attr(spirv),
    no_std
)]

use shared_structs::{MaterialSettings, Uniforms};
use spirv_std::{
    glam::{self, Mat3, Mat4, Vec2, Vec3, Vec4},
    num_traits::Float,
    Image, Sampler,
};

type SampledImage = Image!(2D, type=f32, sampled);

#[spirv(vertex)]
pub fn vertex(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    instance_translation_and_scale: Vec4,
    instance_rotation: glam::Quat,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(position)] builtin_pos: &mut Vec4,
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
) {
    let instance_scale = instance_translation_and_scale.w;
    let instance_translation = instance_translation_and_scale.truncate();

    let position = instance_translation + (instance_rotation * instance_scale * position);
    *builtin_pos = Mat4::from(uniforms.projection_view) * position.extend(1.0);
    builtin_pos.y = -builtin_pos.y;
    *out_position = position;
    *out_normal = instance_rotation * normal;
    *out_uv = uv;
}

struct TextureSampler {
    sampler: Sampler,
    uv: Vec2,
}

impl TextureSampler {
    fn sample(&self, texture: &SampledImage) -> Vec4 {
        texture.sample(self.sampler, self.uv)
    }
}

struct ExtendedMaterialParams {
    base: glam_pbr::MaterialParams,
    alpha: f32,
    emission: Vec3,
}

impl ExtendedMaterialParams {
    pub fn new(
        texture_sampler: &TextureSampler,
        albedo_texture: &SampledImage,
        metallic_roughness_texture: &SampledImage,
        emissive_texture: &SampledImage,
        material_settings: &MaterialSettings,
    ) -> Self {
        let diffuse = texture_sampler.sample(albedo_texture) * material_settings.base_color_factor;
        let emission =
            texture_sampler.sample(emissive_texture).truncate() * material_settings.emissive_factor;

        let metallic_roughness = texture_sampler.sample(metallic_roughness_texture);
        let metallic = metallic_roughness.z * material_settings.metallic_factor;
        let roughness = metallic_roughness.y * material_settings.roughness_factor;

        Self {
            base: glam_pbr::MaterialParams {
                diffuse_colour: diffuse.truncate(),
                metallic,
                perceptual_roughness: glam_pbr::PerceptualRoughness(roughness),
                index_of_refraction: glam_pbr::IndexOfRefraction::default(),
                specular_colour: Vec3::ONE,
                specular_factor: 1.0,
            },
            alpha: diffuse.w,
            emission,
        }
    }
}

#[spirv(fragment)]
pub fn fragment(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 0)] albedo_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 1)] normal_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 2)] metallic_roughness_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 3)] emissive_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 4, uniform)] material_settings: &MaterialSettings,
    output: &mut Vec4,
) {
    let texture_sampler = TextureSampler {
        sampler: *sampler,
        uv,
    };

    let material_params = ExtendedMaterialParams::new(
        &texture_sampler,
        albedo_texture,
        metallic_roughness_texture,
        emissive_texture,
        material_settings,
    );

    let view_vector = (uniforms.eye_position - position).normalize();

    let normal = calculate_normal(normal, uv, view_vector, &texture_sampler, normal_texture);

    let brdf_params = glam_pbr::BasicBrdfParams {
        normal,
        light: glam_pbr::Light(Vec3::ONE.normalize()),
        light_intensity: Vec3::ONE,
        view: glam_pbr::View(view_vector),
        material_params: material_params.base,
    };

    let result = glam_pbr::basic_brdf(brdf_params);

    *output =
        linear_to_srgb(result.diffuse + result.specular + material_params.emission).extend(1.0);
}

#[spirv(fragment)]
pub fn fragment_alpha_clipped(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 0)] albedo_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 1)] normal_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 2)] metallic_roughness_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 3)] emissive_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 4, uniform)] material_settings: &MaterialSettings,
    output: &mut Vec4,
) {
    let texture_sampler = TextureSampler {
        sampler: *sampler,
        uv,
    };

    let material_params = ExtendedMaterialParams::new(
        &texture_sampler,
        albedo_texture,
        metallic_roughness_texture,
        emissive_texture,
        material_settings,
    );

    let view_vector = (uniforms.eye_position - position).normalize();

    let normal = calculate_normal(normal, uv, view_vector, &texture_sampler, normal_texture);

    // We can only do this after we've sampled all textures for naga control flow reasons.
    if material_params.alpha < 0.5 {
        spirv_std::arch::kill();
    }

    let brdf_params = glam_pbr::BasicBrdfParams {
        normal,
        light: glam_pbr::Light(Vec3::ONE.normalize()),
        light_intensity: Vec3::ONE,
        view: glam_pbr::View(view_vector),
        material_params: material_params.base,
    };

    let result = glam_pbr::basic_brdf(brdf_params);

    *output =
        linear_to_srgb(result.diffuse + result.specular + material_params.emission).extend(1.0);
}

fn linear_to_srgb(color_linear: Vec3) -> Vec3 {
    let selector = (color_linear - 0.0031308).ceil(); // 0 if under value, 1 if over
    let under = 12.92 * color_linear;
    let over = 1.055 * color_linear.powf(0.41666) - 0.055;
    under * selector + over * (1.0 - selector)
}

fn calculate_normal(
    interpolated_normal: Vec3,
    uv: Vec2,
    view_vector: Vec3,
    texture_sampler: &TextureSampler,
    normal_map: &SampledImage,
) -> glam_pbr::Normal {
    let normal = interpolated_normal.normalize();

    let map_normal = texture_sampler.sample(normal_map);
    let map_normal = map_normal.truncate();
    let map_normal = map_normal * 255.0 / 127.0 - 128.0 / 127.0;

    let normal = (compute_cotangent_frame(normal, -view_vector, uv) * map_normal).normalize();

    glam_pbr::Normal(normal)
}

fn compute_cotangent_frame(normal: Vec3, position: Vec3, uv: Vec2) -> Mat3 {
    // get edge vectors of the pixel triangle
    let delta_pos_1 = spirv_std::arch::ddx_vector(position);
    let delta_pos_2 = spirv_std::arch::ddy_vector(position);
    let delta_uv_1 = spirv_std::arch::ddx_vector(uv);
    let delta_uv_2 = spirv_std::arch::ddy_vector(uv);

    // solve the linear system
    let delta_pos_2_perp = delta_pos_2.cross(normal);
    let delta_pos_1_perp = normal.cross(delta_pos_1);
    let t = delta_pos_2_perp * delta_uv_1.x + delta_pos_1_perp * delta_uv_2.x;
    let b = delta_pos_2_perp * delta_uv_1.y + delta_pos_1_perp * delta_uv_2.y;

    // construct a scale-invariant frame
    let invmax = 1.0 / t.length_squared().max(b.length_squared()).sqrt();
    Mat3::from_cols(t * invmax, b * invmax, normal)
}

#[spirv(vertex)]
pub fn fullscreen_tri(
    #[spirv(vertex_index)] vert_idx: i32,
    uv: &mut Vec2,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    *uv = Vec2::new(((vert_idx << 1) & 2) as f32, (vert_idx & 2) as f32);
    let mut pos = 2.0 * *uv - Vec2::ONE;
    pos.y = -pos.y;

    *builtin_pos = Vec4::new(pos.x, pos.y, 0.0, 1.0);
}

#[spirv(fragment)]
pub fn blit(
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 1)] texture: &SampledImage,
    output: &mut Vec4,
) {
    *output = texture.sample_by_lod(*sampler, uv, 0.0);
}

#[spirv(vertex)]
pub fn line_vertex(
    position: Vec3,
    colour: Vec3,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(position)] builtin_pos: &mut Vec4,
    out_colour: &mut Vec3,
) {
    *builtin_pos = Mat4::from(uniforms.projection_view) * position.extend(1.0);
    builtin_pos.y = -builtin_pos.y;
    *out_colour = colour;
}

#[spirv(fragment)]
pub fn flat_colour(colour: Vec3, output: &mut Vec4) {
    *output = colour.extend(1.0);
}
