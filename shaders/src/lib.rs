#![cfg_attr(
    target_arch = "spirv",
    feature(register_attr, asm_const, asm_experimental_arch),
    register_attr(spirv),
    no_std
)]

use shared_structs::{MaterialSettings, MirrorUniforms, SkyboxUniforms, Uniforms};
use spirv_std::{
    glam::{self, Mat3, Vec2, Vec3, Vec4},
    num_traits::Float,
    Image, Sampler,
};

type SampledImage = Image!(2D, type=f32, sampled);

mod single_view;

pub use single_view::{
    fragment as _, fragment_alpha_clipped as _, tonemap as _, vertex as _,
    vertex_mirrored as _, vertex_skybox as _, vertex_skybox_mirrored as _,
};

#[spirv(vertex)]
pub fn vertex(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    instance_translation_and_scale: Vec4,
    instance_rotation: glam::Quat,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(position)] builtin_pos: &mut Vec4,
    #[spirv(view_index)] view_index: i32,
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
) {
    let instance_scale = instance_translation_and_scale.w;
    let instance_translation = instance_translation_and_scale.truncate();

    let position = instance_translation + (instance_rotation * instance_scale * position);
    *builtin_pos = uniforms.projection_view(view_index) * position.extend(1.0);
    *out_position = position;
    *out_normal = instance_rotation * normal;
    *out_uv = uv;

    if uniforms.render_direct_to_framebuffer != 0 {
        builtin_pos.y = -builtin_pos.y;
    }
}

struct TextureSampler<'a> {
    sampler: Sampler,
    texture: &'a SampledImage,
    uv: Vec2,
}

impl<'a> TextureSampler<'a> {
    fn new(texture: &'a SampledImage, sampler: Sampler, uv: Vec2) -> Self {
        Self {
            sampler,
            texture,
            uv,
        }
    }

    fn sample(&self) -> Vec4 {
        self.texture.sample(self.sampler, self.uv)
    }
}

struct ExtendedMaterialParams {
    base: glam_pbr::MaterialParams,
    alpha: f32,
    emission: Vec3,
}

impl ExtendedMaterialParams {
    pub fn new(
        albedo_texture: &TextureSampler,
        metallic_roughness_texture: &TextureSampler,
        emissive_texture: &TextureSampler,
        material_settings: &MaterialSettings,
    ) -> Self {
        let albedo = albedo_texture.sample() * material_settings.base_color_factor;
        let emission = emissive_texture.sample().truncate() * material_settings.emissive_factor;

        let metallic_roughness = metallic_roughness_texture.sample();
        let metallic = metallic_roughness.z * material_settings.metallic_factor;
        let roughness = metallic_roughness.y * material_settings.roughness_factor;

        Self {
            base: glam_pbr::MaterialParams {
                albedo_colour: albedo.truncate(),
                metallic,
                perceptual_roughness: glam_pbr::PerceptualRoughness(roughness),
                index_of_refraction: glam_pbr::IndexOfRefraction::default(),
                specular_colour: Vec3::ONE,
                specular_factor: 1.0,
            },
            alpha: albedo.w,
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
    #[spirv(descriptor_set = 0, binding = 2)] ibl_lut: &SampledImage,
    #[spirv(descriptor_set = 0, binding = 3)] diffuse_ibl_cubemap: &Image!(cube, type=f32, sampled),
    #[spirv(descriptor_set = 0, binding = 4)] specular_ibl_cubemap: &Image!(cube, type=f32, sampled),
    #[spirv(descriptor_set = 1, binding = 0)] albedo_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 1)] normal_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 2)] metallic_roughness_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 3)] emissive_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 4, uniform)] material_settings: &MaterialSettings,
    #[spirv(descriptor_set = 1, binding = 5)] albedo_texture_sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 6)] normal_texture_sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 7)] metallic_roughness_texture_sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 8)] emissive_texture_sampler: &Sampler,
    #[spirv(view_index)] view_index: i32,
    #[spirv(front_facing)] front_facing: bool,
    output: &mut Vec4,
) {
    let albedo_texture = TextureSampler::new(albedo_texture, *albedo_texture_sampler, uv);
    let metallic_roughness_texture = TextureSampler::new(
        metallic_roughness_texture,
        *metallic_roughness_texture_sampler,
        uv,
    );
    let normal_texture = TextureSampler::new(normal_texture, *normal_texture_sampler, uv);
    let emissive_texture = TextureSampler::new(emissive_texture, *emissive_texture_sampler, uv);

    let material_params = ExtendedMaterialParams::new(
        &albedo_texture,
        &metallic_roughness_texture,
        &emissive_texture,
        &material_settings,
    );

    if material_settings.is_unlit != 0 {
        *output = material_params.base.albedo_colour.extend(1.0);
        return;
    }

    let view_vector = (uniforms.eye_position(view_index) - position).normalize();

    let normal = calculate_normal(normal, uv, view_vector, &normal_texture, front_facing);
    let view = glam_pbr::View(view_vector);

    let lut_values = glam_pbr::ggx_lut_lookup(
        normal,
        view,
        material_params.base,
        |normal_dot_view: f32, perceptual_roughness: glam_pbr::PerceptualRoughness| {
            let uv = Vec2::new(normal_dot_view, perceptual_roughness.0);
            let sample: Vec4 = ibl_lut.sample_by_lod(*sampler, uv, 0.0);
            Vec2::new(sample.x, sample.y)
        },
    );

    let diffuse_output = glam_pbr::ibl_irradiance_lambertian(
        normal,
        view,
        material_params.base,
        lut_values,
        |normal| {
            let sample: Vec4 = diffuse_ibl_cubemap.sample_by_lod(*sampler, normal, 0.0);
            sample.truncate()
        },
    );

    let specular_output = glam_pbr::get_ibl_radiance_ggx(
        normal,
        view,
        material_params.base,
        lut_values,
        9,
        |ray, lod| {
            let sample: Vec4 = specular_ibl_cubemap.sample_by_lod(*sampler, ray, lod);
            sample.truncate()
        },
    );

    let mut combined_output = diffuse_output + specular_output + material_params.emission;

    if uniforms.inline_tonemapping != 0 {
        combined_output = tonemapping(combined_output);
    }

    *output = combined_output.extend(1.0);
}

#[spirv(fragment)]
pub fn fragment_alpha_clipped(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 2)] ibl_lut: &SampledImage,
    #[spirv(descriptor_set = 0, binding = 3)] diffuse_ibl_cubemap: &Image!(cube, type=f32, sampled),
    #[spirv(descriptor_set = 0, binding = 4)] specular_ibl_cubemap: &Image!(cube, type=f32, sampled),
    #[spirv(descriptor_set = 1, binding = 0)] albedo_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 1)] normal_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 2)] metallic_roughness_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 3)] emissive_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 4, uniform)] material_settings: &MaterialSettings,
    #[spirv(descriptor_set = 1, binding = 5)] albedo_texture_sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 6)] normal_texture_sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 7)] metallic_roughness_texture_sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 8)] emissive_texture_sampler: &Sampler,
    #[spirv(view_index)] view_index: i32,
    #[spirv(front_facing)] front_facing: bool,
    output: &mut Vec4,
) {
    let albedo_texture = TextureSampler::new(albedo_texture, *albedo_texture_sampler, uv);
    let metallic_roughness_texture = TextureSampler::new(
        metallic_roughness_texture,
        *metallic_roughness_texture_sampler,
        uv,
    );
    let normal_texture = TextureSampler::new(normal_texture, *normal_texture_sampler, uv);
    let emissive_texture = TextureSampler::new(emissive_texture, *emissive_texture_sampler, uv);

    let material_params = ExtendedMaterialParams::new(
        &albedo_texture,
        &metallic_roughness_texture,
        &emissive_texture,
        &material_settings,
    );

    let view_vector = (uniforms.eye_position(view_index) - position).normalize();

    let normal = calculate_normal(normal, uv, view_vector, &normal_texture, front_facing);
    let view = glam_pbr::View(view_vector);

    // We can only do this after we've sampled all textures for naga control flow reasons.
    if material_params.alpha < 0.5 {
        spirv_std::arch::kill();
    }

    if material_settings.is_unlit != 0 {
        *output = material_params.base.albedo_colour.extend(1.0);
        return;
    }

    let lut_values = glam_pbr::ggx_lut_lookup(
        normal,
        view,
        material_params.base,
        |normal_dot_view: f32, perceptual_roughness: glam_pbr::PerceptualRoughness| {
            let uv = Vec2::new(normal_dot_view, perceptual_roughness.0);
            let sample: Vec4 = ibl_lut.sample_by_lod(*sampler, uv, 0.0);
            Vec2::new(sample.x, sample.y)
        },
    );

    let diffuse_output = glam_pbr::ibl_irradiance_lambertian(
        normal,
        view,
        material_params.base,
        lut_values,
        |normal| {
            let sample: Vec4 = diffuse_ibl_cubemap.sample_by_lod(*sampler, normal, 0.0);
            sample.truncate()
        },
    );

    let specular_output = glam_pbr::get_ibl_radiance_ggx(
        normal,
        view,
        material_params.base,
        lut_values,
        9,
        |ray, lod| {
            let sample: Vec4 = specular_ibl_cubemap.sample_by_lod(*sampler, ray, lod);
            sample.truncate()
        },
    );

    let mut combined_output = diffuse_output + specular_output + material_params.emission;

    if uniforms.inline_tonemapping != 0 {
        combined_output = tonemapping(combined_output);
    }

    *output = combined_output.extend(1.0);
}

#[spirv(fragment)]
pub fn fragment_ui(
    _position: Vec3,
    _normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 0)] texture: &SampledImage,
    output: &mut Vec4,
) {
    let sample = TextureSampler::new(texture, *sampler, uv).sample();

    let mut colour_output = sample.truncate();

    if uniforms.inline_tonemapping != 0 {
        colour_output = tonemapping(colour_output);
    }

    *output = colour_output.extend(sample.w);
}

fn linear_to_srgb_approx(color_linear: Vec3) -> Vec3 {
    let gamma = 2.2;
    color_linear.powf(1.0 / gamma)
}

fn calculate_normal(
    interpolated_normal: Vec3,
    uv: Vec2,
    view_vector: Vec3,
    normal_map: &TextureSampler,
    front_facing: bool,
) -> glam_pbr::Normal {
    let mut normal = interpolated_normal.normalize();

    if !front_facing {
        normal = -normal;
    }

    let map_normal = normal_map.sample();
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
    #[spirv(vertex_index)] vertex_index: i32,
    uv: &mut Vec2,
    #[spirv(position)] builtin_pos: &mut Vec4,
) {
    *uv = Vec2::new(((vertex_index << 1) & 2) as f32, (vertex_index & 2) as f32);
    let pos = 2.0 * *uv - Vec2::ONE;

    *builtin_pos = Vec4::new(pos.x, pos.y, 0.0, 1.0);
}

#[spirv(fragment)]
pub fn blit(
    mut uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 1)] texture: &SampledImage,
    output: &mut Vec4,
) {
    uv.y = 1.0 - uv.y;
    *output = texture.sample_by_lod(*sampler, uv, 0.0);
}

#[spirv(vertex)]
pub fn vertex_mirrored(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    instance_translation_and_scale: Vec4,
    instance_rotation: glam::Quat,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 2, binding = 0, uniform)] mirror_uniforms: &MirrorUniforms,
    #[spirv(position)] builtin_pos: &mut Vec4,
    #[spirv(view_index)] view_index: i32,
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
) {
    let instance_scale = instance_translation_and_scale.w;
    let instance_translation = instance_translation_and_scale.truncate();

    let position = instance_translation + (instance_rotation * instance_scale * position);
    let position = shared_structs::reflect_in_mirror(
        position,
        mirror_uniforms.position,
        mirror_uniforms.normal,
    );

    let normal = instance_rotation * normal;
    let normal = shared_structs::reflect(normal, mirror_uniforms.normal);

    *builtin_pos = uniforms.projection_view(view_index) * position.extend(1.0);
    *out_position = position;
    *out_normal = normal;
    *out_uv = uv;

    if uniforms.render_direct_to_framebuffer != 0 {
        builtin_pos.y = -builtin_pos.y;
    }
}

// Used for testing stencil writes.
#[spirv(fragment)]
pub fn flat_blue(output: &mut Vec4) {
    *output = Vec4::new(0.0, 0.0, 1.0, 1.0);
}

fn saturate(value: Vec3) -> Vec3 {
    value.max(Vec3::ZERO).min(Vec3::ONE)
}

// https://knarkowicz.wordpress.com/2016/01/06/aces-filmic-tone-mapping-curve/
fn aces_filmic(x: Vec3) -> Vec3 {
    let a = 2.51;
    let b = 0.03;
    let c = 2.43;
    let d = 0.59;
    let e = 0.14;
    saturate((x * (a * x + b)) / (x * (c * x + d) + e))
}

fn tonemapping(hdr_linear: Vec3) -> Vec3 {
    let tonemapped_linear = aces_filmic(hdr_linear);
    linear_to_srgb_approx(tonemapped_linear)
}

#[spirv(fragment)]
pub fn tonemap(
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 1, binding = 0)] sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 1)] texture: &Image!(2D, type=f32, sampled, arrayed),
    output: &mut Vec4,
) {
    let (array_index, uv) = if uv.x > 0.5 {
        (1, Vec2::new(uv.x * 2.0 - 1.0, uv.y))
    } else {
        (0, Vec2::new(uv.x * 2.0, uv.y))
    };

    let sample: Vec4 = texture.sample(*sampler, uv.extend(array_index as f32));

    let mut colour_output = sample.truncate();

    if uniforms.inline_tonemapping == 0 {
        colour_output = tonemapping(colour_output);
    }

    *output = colour_output.extend(1.0)
}

#[spirv(vertex)]
pub fn vertex_skybox(
    #[spirv(vertex_index)] vertex_index: i32,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 1, binding = 0, uniform)] skybox_uniforms: &SkyboxUniforms,
    #[spirv(position)] builtin_pos: &mut Vec4,
    #[spirv(view_index)] view_index: i32,
    ray: &mut Vec3,
) {
    // https://github.com/gfx-rs/wgpu/blob/9114283707a8b472412cf4fe685d364327d3a5b4/wgpu/examples/skybox/shader.wgsl#L21
    let pos = Vec4::new(
        (vertex_index / 2) as f32 * 4.0 - 1.0,
        (vertex_index & 1) as f32 * 4.0 - 1.0,
        1.0,
        1.0,
    );

    let unprojected: Vec4 = skybox_uniforms.projection_inverse(view_index) * pos;

    *ray = glam::Quat::from_vec4(skybox_uniforms.view_inverse(view_index)) * unprojected.truncate();

    *builtin_pos = pos;

    if uniforms.render_direct_to_framebuffer != 0 {
        builtin_pos.y = -builtin_pos.y;
    }
}

#[spirv(vertex)]
pub fn vertex_skybox_mirrored(
    #[spirv(vertex_index)] vertex_index: i32,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 1, binding = 0, uniform)] skybox_uniforms: &SkyboxUniforms,
    #[spirv(descriptor_set = 2, binding = 0, uniform)] mirror_uniforms: &MirrorUniforms,
    #[spirv(position)] builtin_pos: &mut Vec4,
    #[spirv(view_index)] view_index: i32,
    ray: &mut Vec3,
) {
    // https://github.com/gfx-rs/wgpu/blob/9114283707a8b472412cf4fe685d364327d3a5b4/wgpu/examples/skybox/shader.wgsl#L21
    let pos = Vec4::new(
        (vertex_index / 2) as f32 * 4.0 - 1.0,
        (vertex_index & 1) as f32 * 4.0 - 1.0,
        1.0,
        1.0,
    );

    let unprojected: Vec4 = skybox_uniforms.projection_inverse(view_index) * pos;

    *ray = shared_structs::reflect(
        glam::Quat::from_vec4(skybox_uniforms.view_inverse(view_index)) * unprojected.truncate(),
        mirror_uniforms.normal,
    );

    *builtin_pos = pos;

    if uniforms.render_direct_to_framebuffer != 0 {
        builtin_pos.y = -builtin_pos.y;
    }
}

#[spirv(fragment)]
pub fn fragment_skybox(
    ray: Vec3,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 0, binding = 1)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 4)] specular_ibl_cubemap: &Image!(cube, type=f32, sampled),
    output: &mut Vec4,
) {
    let sample: Vec4 = specular_ibl_cubemap.sample_by_lod(*sampler, ray, 0.0);

    let mut skybox_output = sample.truncate();

    if uniforms.inline_tonemapping != 0 {
        skybox_output = tonemapping(skybox_output);
    }

    *output = skybox_output.extend(1.0);
}
