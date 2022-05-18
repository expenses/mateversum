use crate::PerformanceSettings;
use crevice::std140::AsStd140;
use glam::{Vec2, Vec3};
use std::borrow::Cow;
use std::cell::RefCell;
use std::io::Read;
use std::num::NonZeroU32;
use std::ops::Range;
use std::rc::Rc;
use wgpu::util::DeviceExt;

mod deprecated;

use deprecated::*;

use crate::caching::{PipelineData, ResourceCache};

pub(crate) struct ModelLoadContext {
    pub(crate) device: Rc<wgpu::Device>,
    pub(crate) queue: Rc<wgpu::Queue>,
    pub(crate) fetched_images: Rc<RefCell<FetchedImages>>,
    pub(crate) model_bgl: Rc<wgpu::BindGroupLayout>,
    pub(crate) black_image: Rc<Texture>,
    pub(crate) white_image: Rc<Texture>,
    pub(crate) flat_normals_image: Rc<Texture>,
    pub(crate) default_metallic_roughness_image: Rc<Texture>,
    pub(crate) compressed_texture_format: Format,
    pub(crate) shader_cache: Rc<ResourceCache<wgpu::ShaderModule>>,
    pub(crate) pipeline_cache: Rc<ResourceCache<PipelineData>>,
    pub(crate) sampler: Rc<wgpu::Sampler>,
    pub(crate) performance_settings: PerformanceSettings,
    pub(crate) thread_pool: wasm_futures_executor::ThreadPool,
    pub(crate) request_client: RequestClient,
}

struct ModelBuffers {
    map: std::collections::HashMap<usize, Vec<u8>>,
}

struct MaterialTexture {
    texture: Rc<RefCell<Rc<Texture>>>,
    sampler: Rc<RefCell<Rc<wgpu::Sampler>>>,
}

impl MaterialTexture {
    fn new(texture: Rc<Texture>, context: &ModelLoadContext) -> Self {
        Self {
            texture: Rc::new(RefCell::new(texture)),
            sampler: Rc::new(RefCell::new(Rc::clone(&context.sampler))),
        }
    }
}

struct MaterialTextures {
    normal_texture: MaterialTexture,
    albedo_texture: MaterialTexture,
    metallic_roughness_texture: MaterialTexture,
    emissive_texture: MaterialTexture,
}

struct TextureLoadContext {
    gltf: Rc<gltf::Gltf>,
    context: Rc<ModelLoadContext>,
    buffers: Rc<ModelBuffers>,
    textures: Rc<MaterialTextures>,
    material_settings: Rc<wgpu::Buffer>,
    bind_group: Rc<RefCell<wgpu::BindGroup>>,
    base_url: Rc<Option<url::Url>>,
}

async fn upload_model_texture_from_gltf(
    gltf_texture: &gltf::Texture<'_>,
    binding: &MaterialTexture,
    srgb: bool,
    context: &TextureLoadContext,
) -> anyhow::Result<()> {
    let texture = load_image_from_gltf(context, gltf_texture, srgb, binding).await?;

    *binding.texture.borrow_mut() = texture;

    let new_bind_group = create_model_bind_group(
        &context.context,
        &context.textures,
        &context.material_settings,
    );

    *context.bind_group.borrow_mut() = new_bind_group;

    Ok(())
}

fn create_model_bind_group(
    context: &ModelLoadContext,
    textures: &MaterialTextures,
    material_settings: &wgpu::Buffer,
) -> wgpu::BindGroup {
    context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &context.model_bgl,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::TextureView(
                        &textures.albedo_texture.texture.borrow().view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(
                        &textures.normal_texture.texture.borrow().view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(
                        &textures.metallic_roughness_texture.texture.borrow().view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(
                        &textures.emissive_texture.texture.borrow().view,
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: material_settings.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Sampler(
                        &textures.albedo_texture.sampler.borrow(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(
                        &textures.normal_texture.sampler.borrow(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::Sampler(
                        &textures.metallic_roughness_texture.sampler.borrow(),
                    ),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(
                        &textures.emissive_texture.sampler.borrow(),
                    ),
                },
            ],
        })
}

pub(crate) struct ModelPrimitive {
    pub(crate) indices: wgpu::Buffer,
    pub(crate) positions: wgpu::Buffer,
    pub(crate) normals: wgpu::Buffer,
    pub(crate) uvs: wgpu::Buffer,
    pub(crate) bind_group: Rc<RefCell<wgpu::BindGroup>>,
    pub(crate) num_indices: u32,
    // We hold handles onto the used textures here, so that when the model is dropped, the `Rc::strong_count`
    // of the textures goes down. Then we are able to unload the textures from GPU memory by `HashMap::retain`ing the fetched images..
    _textures: Rc<MaterialTextures>,
}

struct StagingModelPrimitive {
    indices: Vec<u32>,
    positions: Vec<Vec3>,
    normals: Vec<Vec3>,
    uvs: Vec<Vec2>,
    material_index: usize,
    material_settings: Rc<wgpu::Buffer>,
}

impl StagingModelPrimitive {
    fn upload(
        self,
        gltf: &Rc<gltf::Gltf>,
        context: &Rc<ModelLoadContext>,
        buffers: &Rc<ModelBuffers>,
        base_url: &Rc<Option<url::Url>>,
    ) -> ModelPrimitive {
        let textures = Rc::new(MaterialTextures {
            albedo_texture: MaterialTexture::new(Rc::clone(&context.white_image), context),
            normal_texture: MaterialTexture::new(Rc::clone(&context.flat_normals_image), context),
            metallic_roughness_texture: MaterialTexture::new(
                Rc::clone(&context.default_metallic_roughness_image),
                context,
            ),
            emissive_texture: MaterialTexture::new(Rc::clone(&context.black_image), context),
        });

        let material_index = self.material_index;
        let material_settings = self.material_settings;

        let bind_group = Rc::new(RefCell::new(create_model_bind_group(
            context,
            &textures,
            &material_settings,
        )));

        let texture_load_context = Rc::new(TextureLoadContext {
            gltf: Rc::clone(gltf),
            context: Rc::clone(context),
            buffers: Rc::clone(buffers),
            textures: Rc::clone(&textures),
            material_settings: Rc::clone(&material_settings),
            bind_group: Rc::clone(&bind_group),
            base_url: Rc::clone(base_url),
        });

        wasm_bindgen_futures::spawn_local({
            let textures = Rc::clone(&textures);
            let context = Rc::clone(&texture_load_context);
            async move {
                if let Some(material) = context.gltf.materials().nth(material_index) {
                    let pbr = material.pbr_metallic_roughness();
                    if let Some(albedo_texture) = pbr.base_color_texture() {
                        upload_model_texture_from_gltf(
                            &albedo_texture.texture(),
                            &textures.albedo_texture,
                            true,
                            &context,
                        )
                        .await
                        .unwrap();
                    }
                } else {
                    log::warn!("Material index is invalid or model contains no materials.")
                }
            }
        });

        wasm_bindgen_futures::spawn_local({
            let textures = Rc::clone(&textures);
            let context = Rc::clone(&texture_load_context);
            async move {
                if let Some(material) = context.gltf.materials().nth(material_index) {
                    if let Some(normal_texture) = material.normal_texture() {
                        upload_model_texture_from_gltf(
                            &normal_texture.texture(),
                            &textures.normal_texture,
                            false,
                            &context,
                        )
                        .await
                        .unwrap();
                    }
                } else {
                    log::warn!("Material index is invalid or model contains no materials.")
                }
            }
        });

        wasm_bindgen_futures::spawn_local({
            let textures = Rc::clone(&textures);
            let context = Rc::clone(&texture_load_context);
            async move {
                if let Some(material) = context.gltf.materials().nth(material_index) {
                    let pbr = material.pbr_metallic_roughness();
                    if let Some(metallic_roughness_texture) = pbr.metallic_roughness_texture() {
                        upload_model_texture_from_gltf(
                            &metallic_roughness_texture.texture(),
                            &textures.metallic_roughness_texture,
                            false,
                            &context,
                        )
                        .await
                        .unwrap();
                    }
                } else {
                    log::warn!("Material index is invalid or model contains no materials.")
                }
            }
        });

        wasm_bindgen_futures::spawn_local({
            let textures = Rc::clone(&textures);
            let context = Rc::clone(&texture_load_context);
            async move {
                if let Some(material) = context.gltf.materials().nth(material_index) {
                    if let Some(emissive_texture) = material.emissive_texture() {
                        upload_model_texture_from_gltf(
                            &emissive_texture.texture(),
                            &textures.emissive_texture,
                            true,
                            &context,
                        )
                        .await
                        .unwrap();
                    }
                } else {
                    log::warn!("Material index is invalid or model contains no materials.")
                }
            }
        });

        ModelPrimitive {
            bind_group,
            num_indices: self.indices.len() as u32,
            indices: context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("indices"),
                    contents: bytemuck::cast_slice(&self.indices),
                    usage: wgpu::BufferUsages::INDEX,
                }),
            positions: context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("positions"),
                    contents: bytemuck::cast_slice(&self.positions),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            normals: context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("normals"),
                    contents: bytemuck::cast_slice(&self.normals),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            uvs: context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("uvs"),
                    contents: bytemuck::cast_slice(&self.uvs),
                    usage: wgpu::BufferUsages::VERTEX,
                }),
            _textures: textures,
        }
    }
}

#[derive(Default)]
pub(crate) struct Model {
    pub(crate) opaque_primitives: Vec<ModelPrimitive>,
    pub(crate) alpha_clipped_primitives: Vec<ModelPrimitive>,
    pub(crate) unlit_opaque_primitives: Vec<ModelPrimitive>,
    pub(crate) unlit_alpha_clipped_primitives: Vec<ModelPrimitive>,
}

pub(crate) async fn load_gltf_from_bytes(
    bytes: &[u8],
    base_url: Option<url::Url>,
    context: &Rc<ModelLoadContext>,
) -> anyhow::Result<Model> {
    let gltf = gltf::Gltf::from_slice(bytes).unwrap();

    let mut buffers = ModelBuffers {
        map: Default::default(),
    };

    let node_tree = gltf_helpers::NodeTree::new(gltf.nodes());

    for buffer in gltf.buffers() {
        match buffer.source() {
            gltf::buffer::Source::Bin => {}
            gltf::buffer::Source::Uri(uri) => {
                let url = url::Url::options()
                    .base_url(base_url.as_ref())
                    .parse(uri)
                    .unwrap();

                if url.scheme() == "data" {
                    let (mime_type, data) = url.path().split_once(',').unwrap();
                    log::info!("Got: {}", mime_type);
                    log::error!("Loading buffers from base64 is deprecated!");
                    buffers
                        .map
                        .insert(buffer.index(), base64::decode(data).unwrap());
                } else {
                    buffers.map.insert(
                        buffer.index(),
                        context.request_client.fetch_bytes(&url, None).await?,
                    );
                }
            }
        }
    }

    let mut opaque_primitives = std::collections::HashMap::new();
    let mut alpha_clipped_primitives = std::collections::HashMap::new();
    let mut unlit_opaque_primitives = std::collections::HashMap::new();
    let mut unlit_alpha_clipped_primitives = std::collections::HashMap::new();

    for (node, mesh) in gltf
        .nodes()
        .filter_map(|node| node.mesh().map(|mesh| (node, mesh)))
    {
        let transform = node_tree.transform_of(node.index());

        for primitive in mesh.primitives() {
            let material = primitive.material();

            let primitive_map = match (material.alpha_mode(), material.unlit()) {
                (gltf::material::AlphaMode::Opaque, false) => &mut opaque_primitives,
                (_, false) => &mut alpha_clipped_primitives,
                (gltf::material::AlphaMode::Opaque, true) => &mut unlit_opaque_primitives,
                (_, true) => &mut unlit_alpha_clipped_primitives,
            };

            // We can't use `or_insert_with` here as that uses a closure and closures aren't async.
            let staging_primitive = match primitive_map.entry(material.index()) {
                std::collections::hash_map::Entry::Occupied(occupied) => occupied.into_mut(),
                std::collections::hash_map::Entry::Vacant(vacancy) => {
                    let pbr = material.pbr_metallic_roughness();

                    vacancy.insert(StagingModelPrimitive {
                        indices: Default::default(),
                        positions: Default::default(),
                        normals: Default::default(),
                        uvs: Default::default(),
                        material_settings: Rc::new(
                            context
                                .device
                                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                                    label: Some("material settings"),
                                    contents: bytemuck::bytes_of(
                                        &shared_structs::MaterialSettings {
                                            base_color_factor: pbr.base_color_factor().into(),
                                            emissive_factor: material.emissive_factor().into(),
                                            metallic_factor: pbr.metallic_factor(),
                                            roughness_factor: pbr.roughness_factor(),
                                        }
                                        .as_std140(),
                                    ),
                                    usage: wgpu::BufferUsages::UNIFORM,
                                }),
                        ),
                        material_index: material.index().unwrap_or(0),
                    })
                }
            };

            let reader = primitive.reader(|buffer| match buffer.source() {
                gltf::buffer::Source::Bin => Some(gltf.blob.as_ref().unwrap()),
                gltf::buffer::Source::Uri(_) => {
                    buffers.map.get(&buffer.index()).map(|vec| &vec[..])
                }
            });

            staging_primitive.indices.extend(
                reader
                    .read_indices()
                    .unwrap()
                    .into_u32()
                    .map(|index| staging_primitive.positions.len() as u32 + index),
            );
            staging_primitive.positions.extend(
                reader
                    .read_positions()
                    .unwrap()
                    .map(|pos| transform * Vec3::from(pos)),
            );
            staging_primitive.normals.extend(
                reader
                    .read_normals()
                    .unwrap()
                    .map(|rot| transform.rotation * Vec3::from(rot)),
            );
            staging_primitive.uvs.extend(
                reader
                    .read_tex_coords(0)
                    .unwrap()
                    .into_f32()
                    .map(glam::Vec2::from),
            );
        }
    }

    let buffers = Rc::new(buffers);
    let gltf = Rc::new(gltf);
    let base_url = Rc::new(base_url);

    Ok(Model {
        opaque_primitives: opaque_primitives
            .into_values()
            .map(|primitive| primitive.upload(&gltf, context, &buffers, &base_url))
            .collect(),
        alpha_clipped_primitives: alpha_clipped_primitives
            .into_values()
            .map(|primitive| primitive.upload(&gltf, context, &buffers, &base_url))
            .collect(),
        unlit_opaque_primitives: unlit_opaque_primitives
            .into_values()
            .map(|primitive| primitive.upload(&gltf, context, &buffers, &base_url))
            .collect(),
        unlit_alpha_clipped_primitives: unlit_alpha_clipped_primitives
            .into_values()
            .map(|primitive| primitive.upload(&gltf, context, &buffers, &base_url))
            .collect(),
    })
}

async fn load_image_from_gltf(
    context: &TextureLoadContext,
    texture: &gltf::Texture<'_>,
    srgb: bool,
    binding: &MaterialTexture,
) -> anyhow::Result<Rc<Texture>> {
    let image = texture.source();

    Ok(match image.source() {
        gltf::image::Source::View { view, mime_type } => {
            log::info!("{} {}", texture.index(), mime_type);
            let buffer = view.buffer();

            let buffer = match buffer.source() {
                gltf::buffer::Source::Bin => context.gltf.blob.as_ref().unwrap(),
                gltf::buffer::Source::Uri(_) => context
                    .buffers
                    .map
                    .get(&buffer.index())
                    .map(|vec| &vec[..])
                    .unwrap(),
            };

            let bytes = &buffer[view.offset()..view.offset() + view.length()];

            log::error!("Loading images from embedded bytes is deprecated!");

            load_image_from_mime_type(
                context,
                ImageSource::Bytes(bytes),
                srgb,
                Some(mime_type),
                binding,
            )
            .await?
        }
        gltf::image::Source::Uri { uri, mime_type } => {
            let url = url::Url::options()
                .base_url(context.base_url.as_ref().as_ref())
                .parse(uri)
                .unwrap();

            if url.scheme() == "data" {
                let (_mime_type, data) = url.path().split_once(',').unwrap();

                log::error!("loading textures from base64 is deprecated!");

                Rc::new(load_standard_image_format(
                    &context.context,
                    &base64::decode(data).unwrap(),
                    srgb,
                ))
            } else {
                if let Some((image, cached_srgb)) =
                    context.context.fetched_images.borrow().get(&url)
                {
                    if *cached_srgb == srgb {
                        return Ok(Rc::clone(image));
                    } else {
                        log::warn!(
                            "Same URL image is used twice, in both srgb and non-srgb formats: {}",
                            url
                        );
                    }
                }

                let url = Rc::new(url);

                let image = load_image_from_mime_type(
                    context,
                    ImageSource::Url(Rc::clone(&url)),
                    srgb,
                    mime_type,
                    binding,
                )
                .await?;

                context
                    .context
                    .fetched_images
                    .borrow_mut()
                    .insert(url, (Rc::clone(&image), srgb));

                image
            }
        }
    })
}

enum ImageSource<'a> {
    Url(Rc<url::Url>),
    Bytes(&'a [u8]),
}

impl<'a> ImageSource<'a> {
    async fn get_bytes(&self, client: &RequestClient) -> anyhow::Result<Cow<'a, [u8]>> {
        Ok(match self {
            Self::Url(url) => Cow::Owned(client.fetch_bytes(url, None).await?),
            Self::Bytes(bytes) => Cow::Borrowed(bytes),
        })
    }
}

async fn load_image_from_mime_type(
    context: &TextureLoadContext,
    source: ImageSource<'_>,
    srgb: bool,
    mime_type: Option<&str>,
    binding: &MaterialTexture,
) -> anyhow::Result<Rc<Texture>> {
    Ok(if mime_type == Some("image/ktx2") {
        match source {
            ImageSource::Bytes(bytes) => Rc::new(load_ktx2_sync(
                &context.context.device,
                &context.context.queue,
                srgb,
                context.context.compressed_texture_format,
                bytes,
            )),
            ImageSource::Url(url) => load_ktx2_async(context, srgb, &url, binding).await?,
        }
    } else if mime_type == Some("image/x.basis") {
        log::error!("Loading .basis files is deprecated!");

        Rc::new(load_basis(
            &context.context.device,
            &context.context.queue,
            context.context.compressed_texture_format,
            &source.get_bytes(&context.context.request_client).await?,
            srgb,
        ))
    } else {
        log::error!("Loading standard jpg/pngs is deprecated!");

        Rc::new(load_standard_image_format(
            &context.context,
            &source.get_bytes(&context.context.request_client).await?,
            srgb,
        ))
    })
}

#[derive(Clone, Copy, Debug)]
pub enum Format {
    Bc7,
    Astc,
    Rgba,
}

impl Format {
    // https://github.com/KhronosGroup/3D-Formats-Guidelines/blob/main/KTXDeveloperGuide.md#primary-transcode-targets
    // suggests we have Astc as 1st priority, Bc7 as second and then fallback to uncompressed rgba.
    pub fn new_from_features(features: wgpu::Features) -> Self {
        if features.contains(wgpu::Features::TEXTURE_COMPRESSION_ASTC_LDR) {
            Self::Astc
        } else if features.contains(wgpu::Features::TEXTURE_COMPRESSION_BC) {
            Self::Bc7
        } else {
            Self::Rgba
        }
    }

    fn as_transcoder_block_format(&self) -> basis_universal::TranscoderBlockFormat {
        match self {
            Self::Bc7 => basis_universal::TranscoderBlockFormat::BC7,
            Self::Astc => basis_universal::TranscoderBlockFormat::ASTC_4x4,
            Self::Rgba => basis_universal::TranscoderBlockFormat::RGBA32,
        }
    }

    fn as_transcoder_format(&self) -> basis_universal::TranscoderTextureFormat {
        match self {
            Self::Bc7 => basis_universal::TranscoderTextureFormat::BC7_RGBA,
            Self::Astc => basis_universal::TranscoderTextureFormat::ASTC_4x4_RGBA,
            Self::Rgba => basis_universal::TranscoderTextureFormat::RGBA32,
        }
    }

    fn as_wgpu(&self, srgb: bool) -> wgpu::TextureFormat {
        match self {
            Self::Astc => wgpu::TextureFormat::Astc {
                block: wgpu::AstcBlock::B4x4,
                channel: if srgb {
                    wgpu::AstcChannel::UnormSrgb
                } else {
                    wgpu::AstcChannel::Unorm
                },
            },
            Self::Rgba => {
                if srgb {
                    wgpu::TextureFormat::Rgba8UnormSrgb
                } else {
                    wgpu::TextureFormat::Rgba8Unorm
                }
            }
            Self::Bc7 => {
                if srgb {
                    wgpu::TextureFormat::Bc7RgbaUnormSrgb
                } else {
                    wgpu::TextureFormat::Bc7RgbaUnorm
                }
            }
        }
    }
}

async fn load_ktx2_async(
    context: &TextureLoadContext,
    srgb: bool,
    url: &Rc<url::Url>,
    binding: &MaterialTexture,
) -> anyhow::Result<Rc<Texture>> {
    // todo:
    // * At the moment it takes 3 round trips to load the first mip:
    //   The header, the level indices and then the first mip data.
    //
    //   This is very slow on high-latency connections. Instead of
    //   using range requests for these 3 pieces, we should instead
    //   request the whole file and use an abort controller
    //   to cancel it: https://javascript.info/fetch-abort.
    //
    // * We could also hide some of the latency while requesting the images
    //   while loading the large geometry blob.

    fn downscaling_for_max_size(texture_size: u32, max_size: u32) -> u32 {
        let texture_size_log = (texture_size as f32).log2();
        let max_size_log = (max_size as f32).log2();

        (texture_size_log as u32).saturating_sub(max_size_log as u32)
    }

    let mut header_bytes = [0; ktx2::Header::LENGTH];

    context
        .context
        .request_client
        .fetch_uint8_array(url, Some(0..ktx2::Header::LENGTH), true)
        .await?
        .copy_to(&mut header_bytes);

    let header = ktx2::Header::from_bytes(&header_bytes);

    header.validate()?;

    let down_scaling_level = context
        .context
        .performance_settings
        .max_texture_size
        .map(|size| downscaling_for_max_size(header.pixel_width.max(header.pixel_width), size))
        .unwrap_or(0)
        .min(header.level_count - 1);

    let mut level_indices = Vec::with_capacity(header.level_count as usize);

    {
        let mut reader = std::io::Cursor::new(
            context
                .context
                .request_client
                .fetch_bytes(
                    url,
                    Some(
                        ktx2::Header::LENGTH
                            ..ktx2::Header::LENGTH
                                + ktx2::LevelIndex::LENGTH * header.level_count as usize,
                    ),
                )
                .await?,
        );

        for _ in 0..header.level_count {
            let mut level_index_bytes = [0; ktx2::LevelIndex::LENGTH];

            reader.read_exact(&mut level_index_bytes)?;

            level_indices.push(ktx2::LevelIndex::from_bytes(&level_index_bytes));
        }
    }

    let format = context.context.compressed_texture_format;

    let downscaled_width = header.pixel_width >> down_scaling_level;
    let downscaled_height = header.pixel_height >> down_scaling_level;

    let texture_descriptor = move || wgpu::TextureDescriptor {
        label: None,
        size: wgpu::Extent3d {
            // Compressed textures made made of 4x4 blocks, so there are some issues
            // with textures that don't have a side length divisible by 4.
            // They're considered fine everywhere except D3D11 and old versions of D3D12
            // (according to jasperrlz in the Wgpu Users element chat).
            width: downscaled_width - (downscaled_width % 4),
            height: downscaled_height - (downscaled_height % 4),
            depth_or_array_layers: 1,
        },
        mip_level_count: header.level_count - down_scaling_level,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: format.as_wgpu(srgb),
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
    };

    // Create a sampler that will only display mip levels from `level` onwards.
    let sampler_descriptor = move |level, context: &ModelLoadContext| wgpu::SamplerDescriptor {
        address_mode_u: wgpu::AddressMode::Repeat,
        address_mode_v: wgpu::AddressMode::Repeat,
        mag_filter: wgpu::FilterMode::Linear,
        min_filter: wgpu::FilterMode::Linear,
        mipmap_filter: wgpu::FilterMode::Linear,
        anisotropy_clamp: context.performance_settings.anisotropy_clamp(),
        lod_min_clamp: (level - down_scaling_level) as f32,
        ..Default::default()
    };

    let texture = Rc::new(Texture::new(
        context.context.device.create_texture(&texture_descriptor()),
    ));

    let mut levels = level_indices.into_iter().enumerate().rev();

    // Load the smallest (1x1 pixel) mip first before returning the texture
    {
        let (i, level_index) = levels.next().unwrap();

        let url = Rc::clone(url);
        let textures = Rc::clone(&context.textures);
        let material_settings = Rc::clone(&context.material_settings);
        let bind_group = Rc::clone(&context.bind_group);
        let context = Rc::clone(&context.context);
        let texture = Rc::clone(&texture);
        let sampler = Rc::clone(&binding.sampler);

        let transcoded = decompress_and_transcode(
            &url,
            i as u32,
            level_index,
            header,
            &context.thread_pool,
            format,
            &context.request_client,
        )
        .await
        .unwrap();

        write_bytes_to_texture(
            &context.queue,
            &texture.texture,
            i as u32 - down_scaling_level,
            &transcoded,
            &texture_descriptor(),
        );

        *sampler.borrow_mut() = Rc::new(
            context
                .device
                .create_sampler(&sampler_descriptor(i as u32, &context)),
        );

        let new_bind_group = create_model_bind_group(&context, &textures, &material_settings);

        *bind_group.borrow_mut() = new_bind_group;
    }

    // Load all other mips in the background.
    wasm_bindgen_futures::spawn_local({
        let url = Rc::clone(url);
        let textures = Rc::clone(&context.textures);
        let material_settings = Rc::clone(&context.material_settings);
        let bind_group = Rc::clone(&context.bind_group);
        let context = Rc::clone(&context.context);
        let texture = Rc::clone(&texture);
        let sampler = Rc::clone(&binding.sampler);

        async move {
            for (i, level_index) in levels {
                if i < down_scaling_level as usize {
                    return;
                }

                let transcoded = decompress_and_transcode(
                    &url,
                    i as u32,
                    level_index,
                    header,
                    &context.thread_pool,
                    format,
                    &context.request_client,
                )
                .await
                .unwrap();

                write_bytes_to_texture(
                    &context.queue,
                    &texture.texture,
                    i as u32 - down_scaling_level,
                    &transcoded,
                    &texture_descriptor(),
                );

                *sampler.borrow_mut() = Rc::new(
                    context
                        .device
                        .create_sampler(&sampler_descriptor(i as u32, &context)),
                );

                let new_bind_group =
                    create_model_bind_group(&context, &textures, &material_settings);

                *bind_group.borrow_mut() = new_bind_group;
            }
        }
    });

    Ok(texture)
}

async fn decompress_and_transcode(
    url: &url::Url,
    level: u32,
    level_index: ktx2::LevelIndex,
    header: ktx2::Header,
    thread_pool: &wasm_futures_executor::ThreadPool,
    format: Format,
    client: &RequestClient,
) -> anyhow::Result<Vec<u8>> {
    let transcoder = basis_universal::LowLevelUastcTranscoder::new();

    let bytes = client
        .fetch_bytes(
            url,
            Some(
                level_index.offset as usize
                    ..(level_index.offset + level_index.length_bytes) as usize,
            ),
        )
        .await?;

    thread_pool
        .spawn(async move {
            let decompressed = match header.supercompression_scheme {
                Some(ktx2::SupercompressionScheme::Zstandard) => {
                    zstd::bulk::decompress(&bytes, level_index.uncompressed_length_bytes as usize)?
                }
                Some(other) => panic!("Unsupported: {:?}", other),
                None => bytes,
            };

            let slice_width = header.pixel_width >> level;
            let slice_height = header.pixel_height >> level;

            let (block_width_pixels, block_height_pixels) = (4, 4);

            let slice_parameters = basis_universal::SliceParametersUastc {
                num_blocks_x: ((slice_width + block_width_pixels - 1) / block_width_pixels).max(1),
                num_blocks_y: ((slice_height + block_height_pixels - 1) / block_height_pixels)
                    .max(1),
                has_alpha: false,
                original_width: slice_width,
                original_height: slice_height,
            };

            transcoder
                .transcode_slice(
                    &decompressed,
                    slice_parameters,
                    basis_universal::DecodeFlags::HIGH_QUALITY,
                    format.as_transcoder_block_format(),
                )
                .map_err(|err| anyhow::anyhow!("Transcoder error: {:?}", err))
        })
        .await?
}

pub(crate) fn load_single_pixel_image(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    format: wgpu::TextureFormat,
    bytes: &[u8; 4],
) -> Rc<Texture> {
    Rc::new(Texture::new(device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: None,
            size: wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
        },
        bytes,
    )))
}

pub(crate) type FetchedImages = std::collections::HashMap<Rc<url::Url>, (Rc<Texture>, bool)>;

#[allow(dead_code)]
pub(crate) fn prune_fetched_images(fetched_images: &mut FetchedImages) -> u32 {
    let mut removed = 0;

    fetched_images.retain(|_, (texture_ref, _)| {
        // Check the strong count. If a model is using the image then
        // it should be 2: the model + the one in this map. If less than 2
        // (it's normally impossible for strong_count to return 1 but w/e)
        // then we can drop it.
        if Rc::strong_count(texture_ref) < 2 {
            removed += 1;
            false
        } else {
            true
        }
    });

    removed
}

fn mip_levels_for_image_size(width: u32, height: u32) -> u32 {
    (width.max(height) as f32).log2() as u32 + 1
}

fn write_bytes_to_texture(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    mip: u32,
    bytes: &[u8],
    desc: &wgpu::TextureDescriptor,
) {
    let format_info = desc.format.describe();

    let mip_size = desc.mip_level_size(mip).unwrap();

    let mip_physical = mip_size.physical_size(desc.format);

    let width_blocks = mip_physical.width / format_info.block_dimensions.0 as u32;
    let height_blocks = mip_physical.height / format_info.block_dimensions.1 as u32;

    let bytes_per_row = width_blocks * format_info.block_size as u32;

    queue.write_texture(
        wgpu::ImageCopyTexture {
            texture,
            mip_level: mip,
            origin: wgpu::Origin3d::ZERO,
            aspect: wgpu::TextureAspect::All,
        },
        bytes,
        wgpu::ImageDataLayout {
            offset: 0,
            bytes_per_row: Some(NonZeroU32::new(bytes_per_row).expect("invalid bytes per row")),
            rows_per_image: Some(NonZeroU32::new(height_blocks).expect("invalid height")),
        },
        mip_physical,
    );
}

pub(crate) struct Texture {
    pub(crate) texture: wgpu::Texture,
    pub(crate) view: wgpu::TextureView,
}

impl Texture {
    fn new(texture: wgpu::Texture) -> Self {
        Self {
            view: texture.create_view(&Default::default()),
            texture,
        }
    }
}

async fn resolve_promise(promise: js_sys::Promise) -> anyhow::Result<wasm_bindgen::JsValue> {
    wasm_bindgen_futures::JsFuture::from(promise)
        .await
        .map_err(|err| anyhow::anyhow!("{:?}", err))
}

fn byte_range_string(range: Range<usize>) -> String {
    format!("bytes={}-{}", range.start, range.end - 1)
}

fn construct_request_init(
    byte_range: Option<Range<usize>>,
) -> anyhow::Result<web_sys::RequestInit> {
    let mut request_init = web_sys::RequestInit::new();

    if let Some(byte_range) = byte_range {
        let headers = js_sys::Object::new();
        js_sys::Reflect::set(
            &headers,
            &"Range".into(),
            &byte_range_string(byte_range).into(),
        )
        .map_err(|err| anyhow::anyhow!("Js Error: {:?}", err))?;
        request_init.headers(&headers);
    }

    Ok(request_init)
}

pub(crate) struct RequestClient {
    cache: web_sys::Cache,
    ipfs_gateway_url: url::Url,
}

impl RequestClient {
    pub fn new(cache: web_sys::Cache) -> anyhow::Result<Self> {
        Ok(Self {
            cache,
            ipfs_gateway_url: url::Url::parse("http://localhost:8080/ipfs")?,
        })
    }

    async fn fetch_uint8_array(
        &self,
        url: &url::Url,
        byte_range: Option<Range<usize>>,
        cache: bool,
    ) -> anyhow::Result<js_sys::Uint8Array> {
        let request_init = construct_request_init(byte_range.clone())?;

        let mut cache_url = url.clone();

        if let Some(byte_range) = byte_range.clone() {
            cache_url.query_pairs_mut().append_pair(
                "bytes",
                &format!("{}-{}", byte_range.start, byte_range.end - 1),
            );
        }

        let mut fetch_url = url.clone();

        if url.scheme() == "ipfs" {
            fetch_url = self.ipfs_gateway_url.clone();

            let host_err = || anyhow::anyhow!("Failed to get url host");
            let path_segments_err = || anyhow::anyhow!("Failed to get url path segments");
            let scheme_err = || anyhow::anyhow!("Failed to set scheme");

            fetch_url
                .path_segments_mut()
                .map_err(|_| path_segments_err())?
                .extend(
                    std::iter::once(url.host_str().ok_or_else(host_err)?)
                        .chain(url.path_segments().into_iter().flatten()),
                );

            // The Web Cache API only lets you cache http:// or https:// urls.
            // As a result, we need to rewrite the url to:
            // http://ipfs/<CID>/<PATH>

            cache_url
                .set_scheme_unchecked("http")
                .map_err(|_| scheme_err())?;

            // Append the host_str (the CID in this case) to the path.
            let new_path: Vec<_> = std::iter::once(cache_url.host_str().ok_or_else(host_err)?)
                .chain(cache_url.path_segments().into_iter().flatten())
                .map(|string| string.to_owned())
                .collect();
            cache_url
                .path_segments_mut()
                .map_err(|_| path_segments_err())?
                .clear()
                .extend(new_path);

            cache_url.set_host(Some("ipfs"))?;
        }

        let cache_request =
            web_sys::Request::new_with_str_and_init(cache_url.as_str(), &request_init)
                .map_err(|err| anyhow::anyhow!("{:?}", err))?;

        let response = match self.lookup(&cache_request).await? {
            Some(response) => response,
            None => {
                let request =
                    web_sys::Request::new_with_str_and_init(fetch_url.as_str(), &request_init)
                        .map_err(|err| anyhow::anyhow!("{:?}", err))?;

                let response: web_sys::Response =
                    resolve_promise(web_sys::window().unwrap().fetch_with_request(&request))
                        .await?
                        .into();

                if !response.ok() {
                    return Err(anyhow::anyhow!(
                        "Bad fetch response:\nGot status code {} for {}",
                        response.status(),
                        url
                    ));
                }

                let response = if byte_range.is_some() {
                    let array_buffer: js_sys::ArrayBuffer = resolve_promise(
                        response
                            .array_buffer()
                            .map_err(|err| anyhow::anyhow!("{:?}", err))?,
                    )
                    .await?
                    .into();

                    let mut response_init = web_sys::ResponseInit::new();

                    response_init.headers(&response.headers());

                    let fabricated_response =
                        web_sys::Response::new_with_opt_buffer_source_and_init(
                            Some(&array_buffer.into()),
                            &response_init,
                        );

                    fabricated_response.map_err(|err| anyhow::anyhow!("{:?}", err))?
                } else {
                    response
                };

                if cache {
                    self.store(&cache_request, &response).await?;

                    self.lookup(&cache_request).await?.unwrap()
                } else {
                    response
                }
            }
        };

        let array_buffer: js_sys::ArrayBuffer = resolve_promise(
            response
                .array_buffer()
                .map_err(|err| anyhow::anyhow!("{:?}", err))?,
        )
        .await?
        .into();

        let uint8_array = js_sys::Uint8Array::new(&array_buffer);

        Ok(uint8_array)
    }

    pub async fn fetch_bytes(
        &self,
        url: &url::Url,
        byte_range: Option<Range<usize>>,
    ) -> anyhow::Result<Vec<u8>> {
        let uint8_array = self.fetch_uint8_array(url, byte_range, true).await?;

        Ok(uint8_array.to_vec())
    }

    pub async fn fetch_bytes_without_caching(
        &self,
        url: &url::Url,
        byte_range: Option<Range<usize>>,
    ) -> anyhow::Result<Vec<u8>> {
        let uint8_array = self.fetch_uint8_array(url, byte_range, false).await?;

        Ok(uint8_array.to_vec())
    }

    async fn store(
        &self,
        request: &web_sys::Request,
        response: &web_sys::Response,
    ) -> anyhow::Result<()> {
        resolve_promise(self.cache.put_with_request(request, response)).await?;

        Ok(())
    }

    async fn lookup(
        &self,
        request: &web_sys::Request,
    ) -> anyhow::Result<Option<web_sys::Response>> {
        let cache_lookup = resolve_promise(self.cache.match_with_request(request)).await?;

        Ok(if !cache_lookup.is_undefined() {
            Some(cache_lookup.into())
        } else {
            None
        })
    }
}
