#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr, asm_const, asm_experimental_arch),
    register_attr(spirv),
    no_std
)]

use shared_structs::Uniforms;
use spirv_std::{
    glam::{Mat3, Mat4, Vec2, Vec3, Vec4},
    num_traits::Float,
    Image, Sampler,
};

type SampledImage = Image!(2D, type=f32, sampled);

#[spirv(vertex)]
pub fn vertex(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(position)] builtin_pos: &mut Vec4,
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
) {
    *builtin_pos = Mat4::from(uniforms.projection_view) * position.extend(1.0);
    builtin_pos.y = -builtin_pos.y;
    *out_position = position;
    *out_normal = normal;
    *out_uv = uv;
}

#[spirv(fragment)]
pub fn fragment(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] albedo_texture: &SampledImage,
    #[spirv(descriptor_set = 0, binding = 3)] normal_texture: &SampledImage,
    #[spirv(descriptor_set = 0, binding = 4)] metallic_roughness_texture: &SampledImage,
    #[spirv(descriptor_set = 0, binding = 5)] emissive_texture: &SampledImage,
    output: &mut Vec4,
) {
    let diffuse: Vec4 = albedo_texture.sample(*sampler, uv);
    let emission: Vec4 = emissive_texture.sample(*sampler, uv);
    let emission = emission.truncate();
    let metallic_roughness: Vec4 = metallic_roughness_texture.sample(*sampler, uv);
    let metallic = metallic_roughness.z;
    let roughness = metallic_roughness.y;

    let view_vector = (uniforms.eye_position - position).normalize();

    let normal = calculate_normal(normal, uv, view_vector, sampler, normal_texture);

    let material_params = glam_pbr::MaterialParams {
        diffuse_colour: diffuse.truncate(),
        metallic,
        perceptual_roughness: glam_pbr::PerceptualRoughness(roughness),
        index_of_refraction: glam_pbr::IndexOfRefraction::default(),
        specular_colour: Vec3::ONE,
        specular_factor: 1.0,
    };

    let brdf_params = glam_pbr::BasicBrdfParams {
        normal,
        light: glam_pbr::Light(Vec3::ONE.normalize()),
        light_intensity: Vec3::ONE,
        view: glam_pbr::View(view_vector),
        material_params,
    };

    let result = glam_pbr::basic_brdf(brdf_params);

    *output = linear_to_srgb(result.diffuse + result.specular + emission).extend(1.0);
}

#[spirv(fragment)]
pub fn fragment_alpha_clipped(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] albedo_texture: &SampledImage,
    #[spirv(descriptor_set = 0, binding = 3)] normal_texture: &SampledImage,
    #[spirv(descriptor_set = 0, binding = 4)] metallic_roughness_texture: &SampledImage,
    #[spirv(descriptor_set = 0, binding = 5)] emissive_texture: &SampledImage,
    output: &mut Vec4,
) {
    let diffuse: Vec4 = albedo_texture.sample(*sampler, uv);

    let emission: Vec4 = emissive_texture.sample(*sampler, uv);
    let emission = emission.truncate();
    let metallic_roughness: Vec4 = metallic_roughness_texture.sample(*sampler, uv);
    let metallic = metallic_roughness.z;
    let roughness = metallic_roughness.y;

    let view_vector = (uniforms.eye_position - position).normalize();

    let normal = calculate_normal(normal, uv, view_vector, sampler, normal_texture);

    let material_params = glam_pbr::MaterialParams {
        diffuse_colour: diffuse.truncate(),
        metallic,
        perceptual_roughness: glam_pbr::PerceptualRoughness(roughness),
        index_of_refraction: glam_pbr::IndexOfRefraction::default(),
        specular_colour: Vec3::ONE,
        specular_factor: 1.0,
    };

    let brdf_params = glam_pbr::BasicBrdfParams {
        normal,
        light: glam_pbr::Light(Vec3::ONE.normalize()),
        light_intensity: Vec3::ONE,
        view: glam_pbr::View(view_vector),
        material_params,
    };

    let result = glam_pbr::basic_brdf(brdf_params);

    if diffuse.w < 0.5 {
        spirv_std::arch::kill();
    }

    *output = linear_to_srgb(result.diffuse + result.specular + emission).extend(1.0);
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
    sampler: &Sampler,
    normal_map: &SampledImage,
) -> glam_pbr::Normal {
    let normal = interpolated_normal.normalize();

    let map_normal: Vec4 = normal_map.sample(*sampler, uv);
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
    let pos = 2.0 * *uv - Vec2::ONE;

    *builtin_pos = Vec4::new(pos.x, pos.y, 0.0, 1.0);
}

#[spirv(fragment)]
pub fn blit(
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 1)] texture: &SampledImage,
    output: &mut Vec4,
) {
    *output = texture.sample(*sampler, uv);
}
