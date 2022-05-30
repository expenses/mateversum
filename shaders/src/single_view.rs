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
    super::vertex(
        position,
        normal,
        uv,
        instance_translation_and_scale,
        instance_rotation,
        uniforms,
        builtin_pos,
        0,
        out_position,
        out_normal,
        out_uv,
    );
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
    output: &mut Vec4,
) {
    super::fragment(
        position,
        normal,
        uv,
        uniforms,
        sampler,
        ibl_lut,
        diffuse_ibl_cubemap,
        specular_ibl_cubemap,
        albedo_texture,
        normal_texture,
        metallic_roughness_texture,
        emissive_texture,
        material_settings,
        albedo_texture_sampler,
        normal_texture_sampler,
        metallic_roughness_texture_sampler,
        emissive_texture_sampler,
        0,
        output,
    );
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
    super::fragment_alpha_clipped(
        position,
        normal,
        uv,
        uniforms,
        _sampler,
        albedo_texture,
        normal_texture,
        metallic_roughness_texture,
        emissive_texture,
        material_settings,
        albedo_texture_sampler,
        normal_texture_sampler,
        metallic_roughness_texture_sampler,
        emissive_texture_sampler,
        0,
        output,
    );
}

#[spirv(vertex)]
pub fn line_vertex(
    position: Vec3,
    colour: Vec3,
    #[spirv(descriptor_set = 0, binding = 0, uniform)] uniforms: &Uniforms,
    #[spirv(position)] builtin_pos: &mut Vec4,
    out_colour: &mut Vec3,
) {
    super::line_vertex(position, colour, uniforms, builtin_pos, 0, out_colour);
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
    super::vertex_mirrored(
        position,
        normal,
        uv,
        instance_translation_and_scale,
        instance_rotation,
        uniforms,
        mirror_uniforms,
        builtin_pos,
        0,
        out_position,
        out_normal,
        out_uv,
    )
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

    *output = linear_to_srgb_approx(linear).extend(1.0)
}
