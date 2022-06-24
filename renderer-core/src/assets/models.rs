use super::HttpClient;
use glam::{Vec2, Vec3};
use std::collections::HashMap;
use std::ops::Range;
use std::sync::Arc;

pub struct Context<T: HttpClient + 'static> {
    pub http_client: T,
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub vertex_buffers: Arc<parking_lot::Mutex<crate::buffers::VertexBuffers>>,
    pub index_buffer: Arc<parking_lot::Mutex<crate::buffers::IndexBuffer>>,
}

#[derive(Default)]
pub struct PrimitiveRanges {
    pub opaque: Range<usize>,
    pub alpha_clipped: Range<usize>,
    pub opaque_double_sided: Range<usize>,
    pub alpha_clipped_double_sided: Range<usize>,
}

#[derive(Default)]
pub struct Model {
    pub primitives: Vec<Primitive>,
    pub primitive_ranges: PrimitiveRanges,
    pub index_buffer_range: Range<u32>,
    pub vertex_buffer_range: Range<u32>,
}

impl Model {
    pub async fn load<T: HttpClient>(
        context: &Context<T>,
        root_url: &url::Url,
    ) -> anyhow::Result<Model> {
        let gltf: gltf::Gltf<()> =
            gltf::Gltf::from_slice(&context.http_client.fetch_bytes(root_url, None).await?)?;

        let mut buffer_map = HashMap::new();

        let node_tree = gltf_helpers::NodeTree::new(gltf.nodes());

        for buffer in gltf.buffers() {
            match buffer.source() {
                gltf::buffer::Source::Bin => {}
                gltf::buffer::Source::Uri(uri) => {
                    let url = url::Url::options().base_url(Some(root_url)).parse(uri)?;

                    if url.scheme() == "data" {
                        let (mime_type, data) = url.path().split_once(',').unwrap();
                        log::warn!("Loading buffers from embedded base64 is inefficient. Consider moving the buffers into a seperate file.");
                        buffer_map.insert(buffer.index(), base64::decode(data)?);
                    } else {
                        buffer_map.insert(
                            buffer.index(),
                            context.http_client.fetch_bytes(&url, None).await?,
                        );
                    }
                }
            }
        }

        // What we're doing here is essentially collecting all the model primitives that share a meterial together
        // to reduce the number of draw calls.
        let mut opaque_primitives = HashMap::new();
        let mut alpha_clipped_primitives = HashMap::new();
        let mut opaque_double_sided_primitives = HashMap::new();
        let mut alpha_clipped_double_sided_primitives = HashMap::new();

        for (node, mesh) in gltf
            .nodes()
            .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
        {
            let transform = node_tree.transform_of(node.index());

            for primitive in mesh.primitives() {
                let material = primitive.material();

                // Note: it's possible to render double-sided objects with a backface-culling shader if we double the
                // triangles in the index buffer but with a backwards winding order. It's only worth doing this to keep
                // the number of shader permutations down.
                //
                // One thing to keep in mind is that we flip the shading normals according to the gltf spec:
                // https://www.khronos.org/registry/glTF/specs/2.0/glTF-2.0.html#double-sided

                let primitive_map = match (material.alpha_mode(), material.double_sided()) {
                    (gltf::material::AlphaMode::Opaque, false) => &mut opaque_primitives,
                    (_, false) => &mut alpha_clipped_primitives,
                    (gltf::material::AlphaMode::Opaque, true) => {
                        &mut opaque_double_sided_primitives
                    }
                    (_, true) => &mut alpha_clipped_double_sided_primitives,
                };

                let reader = primitive.reader(|buffer| match buffer.source() {
                    gltf::buffer::Source::Bin => Some(gltf.blob.as_ref().unwrap()),
                    gltf::buffer::Source::Uri(_) => {
                        buffer_map.get(&buffer.index()).map(|vec| &vec[..])
                    }
                });

                // Workaround for some exporters (Scaniverse) exporting scanned models that are meant to be
                // rendered unlit but don't set the material flag.
                let unlit = material.unlit() || reader.read_normals().is_none();

                let staging_primitive =
                    primitive_map.entry(material.index()).or_insert_with(|| {
                        let pbr = material.pbr_metallic_roughness();

                        StagingPrimitive {
                            buffers: StagingBuffers::default(),
                            material_settings: shared_structs::MaterialSettings {
                                base_color_factor: pbr.base_color_factor().into(),
                                emissive_factor: material.emissive_factor().into(),
                                metallic_factor: pbr.metallic_factor(),
                                roughness_factor: pbr.roughness_factor(),
                                is_unlit: unlit as u32,
                            },
                            material_index: material.index().unwrap_or(0),
                        }
                    });

                staging_primitive.buffers.indices.extend(
                    reader
                        .read_indices()
                        .unwrap()
                        .into_u32()
                        .map(|index| staging_primitive.buffers.positions.len() as u32 + index),
                );

                let start_positions = staging_primitive.buffers.positions.len();

                staging_primitive.buffers.positions.extend(
                    reader
                        .read_positions()
                        .unwrap()
                        .map(|pos| transform * Vec3::from(pos)),
                );

                let num_positions = staging_primitive.buffers.positions.len() - start_positions;

                match reader.read_normals() {
                    Some(normals) => staging_primitive
                        .buffers
                        .normals
                        .extend(normals.map(|normal| transform.rotation * Vec3::from(normal))),
                    None => staging_primitive
                        .buffers
                        .normals
                        .extend(std::iter::repeat(Vec3::ZERO).take(num_positions)),
                }
                staging_primitive.buffers.uvs.extend(
                    reader
                        .read_tex_coords(0)
                        .unwrap()
                        .into_f32()
                        .map(glam::Vec2::from),
                );
            }
        }

        fn collect_primitives<'a, T: std::iter::Iterator<Item = &'a StagingPrimitive>>(
            primitives: &mut Vec<Primitive>,
            staging_buffers: &mut StagingBuffers,
            staging_primitives: T,
        ) -> Range<usize> {
            let primitives_start = primitives.len();

            for staging_primitive in staging_primitives {
                primitives.push(Primitive {
                    index_buffer_range: staging_buffers.collect(&staging_primitive.buffers),
                });
            }

            let primitives_end = primitives.len();

            primitives_start..primitives_end
        }

        // Collect all the buffers for the primitives into one big staging buffer
        // and collect all the primitive ranges into one big vector.

        let mut staging_buffers = StagingBuffers::default();

        let mut primitives = Vec::new();

        let primitive_ranges = PrimitiveRanges {
            opaque: collect_primitives(
                &mut primitives,
                &mut staging_buffers,
                opaque_primitives.values(),
            ),
            alpha_clipped: collect_primitives(
                &mut primitives,
                &mut staging_buffers,
                alpha_clipped_primitives.values(),
            ),
            opaque_double_sided: collect_primitives(
                &mut primitives,
                &mut staging_buffers,
                opaque_double_sided_primitives.values(),
            ),
            alpha_clipped_double_sided: collect_primitives(
                &mut primitives,
                &mut staging_buffers,
                alpha_clipped_double_sided_primitives.values(),
            ),
        };

        let mut command_encoder =
            context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("command encoder"),
                });

        let vertex_buffer_range = context.vertex_buffers.lock().insert(
            &staging_buffers.positions,
            &staging_buffers.normals,
            &staging_buffers.uvs,
            &context.device,
            &context.queue,
            &mut command_encoder,
        );

        // Make sure the indices point to the right vertices.
        for index in &mut staging_buffers.indices {
            *index += vertex_buffer_range.start;
        }

        let index_buffer_range = context.index_buffer.lock().insert(
            &staging_buffers.indices,
            &context.device,
            &context.queue,
            &mut command_encoder,
        );

        context
            .queue
            .submit(std::iter::once(command_encoder.finish()));

        // Make sure the primitive index ranges are absolute from the start of the buffer.
        for primitive in &mut primitives {
            primitive.index_buffer_range.start += index_buffer_range.start;
            primitive.index_buffer_range.end += index_buffer_range.start;
        }

        Ok(Model {
            primitives,
            primitive_ranges,
            index_buffer_range,
            vertex_buffer_range,
        })
    }
}

struct StagingPrimitive {
    buffers: StagingBuffers,
    material_settings: shared_structs::MaterialSettings,
    material_index: usize,
}

pub struct Primitive {
    pub index_buffer_range: Range<u32>,
}

#[derive(Default)]
struct StagingBuffers {
    indices: Vec<u32>,
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
}

impl StagingBuffers {
    fn collect(&mut self, new: &StagingBuffers) -> Range<u32> {
        let indices_start = self.indices.len() as u32;
        let num_vertices = self.positions.len() as u32;

        self.indices
            .extend(new.indices.iter().map(|index| index + num_vertices));

        self.positions.extend_from_slice(&new.positions);
        self.normals.extend_from_slice(&new.normals);
        self.uvs.extend_from_slice(&new.uvs);

        let indices_end = self.indices.len() as u32;

        indices_start..indices_end
    }
}
