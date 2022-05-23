use super::*;

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
    *builtin_pos = uniforms.projection_view(0) * position.extend(1.0);
    *out_position = position;
    *out_normal = instance_rotation * normal;
    *out_uv = uv;
}

#[spirv(fragment)]
pub fn fragment(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 0, binding = 1)] _sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 0)] albedo_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 1)] normal_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 2)] metallic_roughness_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 3)] emissive_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 4, uniform)] material_settings: &MaterialSettings,
    #[spirv(descriptor_set = 1, binding = 5)] albedo_texture_sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 6)] normal_texture_sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 7)] metallic_roughness_texture_sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 8)] emissive_texture_sampler: &Sampler,
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

    let view_vector = (uniforms.eye_position(0) - position).normalize();

    let normal = calculate_normal(normal, uv, view_vector, &normal_texture);

    let brdf_params = glam_pbr::BasicBrdfParams {
        normal,
        light: glam_pbr::Light(Vec3::ONE.normalize()),
        light_intensity: Vec3::ONE,
        view: glam_pbr::View(view_vector),
        material_params: material_params.base,
    };

    let result = glam_pbr::basic_brdf(brdf_params);

    *output = (result.diffuse + result.specular + material_params.emission).extend(1.0);
}

#[spirv(fragment)]
pub fn fragment_alpha_clipped(
    position: Vec3,
    normal: Vec3,
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(descriptor_set = 0, binding = 1)] _sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 0)] albedo_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 1)] normal_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 2)] metallic_roughness_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 3)] emissive_texture: &SampledImage,
    #[spirv(descriptor_set = 1, binding = 4, uniform)] material_settings: &MaterialSettings,
    #[spirv(descriptor_set = 1, binding = 5)] albedo_texture_sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 6)] normal_texture_sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 7)] metallic_roughness_texture_sampler: &Sampler,
    #[spirv(descriptor_set = 1, binding = 8)] emissive_texture_sampler: &Sampler,
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

    let view_vector = (uniforms.eye_position(0) - position).normalize();

    let normal = calculate_normal(normal, uv, view_vector, &normal_texture);

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

    *output = (result.diffuse + result.specular + material_params.emission).extend(1.0);
}

#[spirv(vertex)]
pub fn line_vertex(
    position: Vec3,
    colour: Vec3,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(position)] builtin_pos: &mut Vec4,
    out_colour: &mut Vec3,
) {
    *builtin_pos = uniforms.projection_view(0) * position.extend(1.0);
    *out_colour = colour;
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
    out_position: &mut Vec3,
    out_normal: &mut Vec3,
    out_uv: &mut Vec2,
) {
    let instance_scale = instance_translation_and_scale.w;
    let instance_translation = instance_translation_and_scale.truncate();

    let position = instance_translation + (instance_rotation * instance_scale * position);
    *builtin_pos = uniforms.projection_view(0)
        * shared_structs::reflect_in_mirror(
            position,
            mirror_uniforms.position,
            mirror_uniforms.normal,
        )
        .extend(1.0);
    *out_position = position;
    *out_normal = instance_rotation * normal;
    *out_uv = uv;
}

#[spirv(fragment)]
pub fn tonemap(
    uv: Vec2,
    #[spirv(descriptor_set = 0, binding = 0)] sampler: &Sampler,
    #[spirv(descriptor_set = 0, binding = 1)] texture: &Image!(2D, type=f32, sampled),
    output: &mut Vec4,
) {
    let sample: Vec4 = texture.sample(*sampler, uv);

    let linear = aces_filmic(sample.truncate());

    *output = linear_to_srgb(linear).extend(1.0)
}
