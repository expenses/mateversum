use arc_swap::ArcSwap;
use std::mem::size_of;
use std::ops::Range;
use std::sync::Arc;

use super::Instance;
use glam::{Vec2, Vec3};

pub type InstanceBuffer = VecGpuBuffer<Instance>;

pub struct VecGpuBuffer<T: bytemuck::Pod> {
    offset: u32,
    capacity: u32,
    pub buffer: wgpu::Buffer,
    usage: wgpu::BufferUsages,
    _phantom: std::marker::PhantomData<T>,
    label: &'static str,
}

impl<T: bytemuck::Pod> VecGpuBuffer<T> {
    fn size_in_bytes(size: u32) -> u64 {
        size as u64 * size_of::<T>() as u64
    }

    pub fn new(
        capacity: u32,
        device: &wgpu::Device,
        usage: wgpu::BufferUsages,
        label: &'static str,
    ) -> Self {
        Self {
            offset: Default::default(),
            capacity,
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: Self::size_in_bytes(capacity),
                usage: usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            }),
            usage,
            label,
            _phantom: Default::default(),
        }
    }

    pub fn clear(&mut self) {
        self.offset = 0;
    }

    pub fn push(
        &mut self,
        instances: &[T],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        command_encoder: &mut wgpu::CommandEncoder,
    ) -> Range<u32> {
        let start = self.offset;
        let end = start + instances.len() as u32;

        if end > self.capacity {
            self.resize(end, device, command_encoder);
        }

        queue.write_buffer(
            &self.buffer,
            Self::size_in_bytes(start),
            bytemuck::cast_slice(instances),
        );

        self.offset = end;

        start..end
    }

    fn resize(
        &mut self,
        required_capacity: u32,
        device: &wgpu::Device,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        let copy_size = Self::size_in_bytes(self.offset);

        let new_capacity = required_capacity.max(self.capacity * 2);

        log::info!(
            "Growing {} from {} to {}",
            self.label,
            self.capacity,
            new_capacity
        );

        let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(self.label),
            size: Self::size_in_bytes(new_capacity),
            usage: self.usage | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        command_encoder.copy_buffer_to_buffer(&self.buffer, 0, &new_buffer, 0, copy_size);

        self.buffer = new_buffer;
        self.capacity = new_capacity;
    }
}

pub struct IndexBuffer {
    allocator: parking_lot::Mutex<range_alloc::RangeAllocator<u32>>,
    pub buffer: ArcSwap<wgpu::Buffer>,
}

impl IndexBuffer {
    fn size_in_bytes(size: u32) -> u64 {
        size as u64 * size_of::<u32>() as u64
    }

    pub fn new(capacity: u32, device: &wgpu::Device) -> Self {
        Self {
            allocator: parking_lot::Mutex::new(range_alloc::RangeAllocator::new(0..capacity)),
            buffer: ArcSwap::from(Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("index buffer"),
                size: Self::size_in_bytes(capacity),
                usage: wgpu::BufferUsages::INDEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }))),
        }
    }

    pub(crate) fn insert(
        &self,
        indices: &[u32],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        command_encoder: &mut wgpu::CommandEncoder,
    ) -> Range<u32> {
        let length = indices.len() as u32;

        // Use the allocator to find a range in the buffer to write to,
        // resizing the buffer in needed and returning the correct buffer to write to
        // (as `ArcSwap::load` does not always return the newest value).
        let (buffer, range) = {
            let mut allocator = self.allocator.lock();

            match allocator.allocate_range(length) {
                Ok(range) => (self.buffer.load_full(), range),
                Err(_) => {
                    let new_buffer = Self::resize(
                        &mut allocator,
                        &self.buffer,
                        length,
                        device,
                        command_encoder,
                    );
                    let range = allocator.allocate_range(length).expect("just resized");
                    (new_buffer, range)
                }
            }
        };

        queue.write_buffer(
            &buffer,
            Self::size_in_bytes(range.start),
            bytemuck::cast_slice(indices),
        );

        range
    }

    fn resize(
        allocator: &mut range_alloc::RangeAllocator<u32>,
        buffer: &ArcSwap<wgpu::Buffer>,
        required_capacity: u32,
        device: &wgpu::Device,
        command_encoder: &mut wgpu::CommandEncoder,
    ) -> Arc<wgpu::Buffer> {
        let copy_range = allocator
            .allocated_ranges()
            .last()
            .map(|range| range.end)
            .unwrap_or(0);

        let old_capacity = allocator.initial_range().end;

        let new_capacity = (old_capacity + required_capacity).max(old_capacity * 2);

        log::info!(
            "Growing index buffer from {} to {}",
            old_capacity,
            new_capacity
        );

        allocator.grow_to(new_capacity);

        let new_buffer = Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("index buffer"),
            size: Self::size_in_bytes(new_capacity),
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }));

        command_encoder.copy_buffer_to_buffer(
            &buffer.load(),
            0,
            &new_buffer,
            0,
            Self::size_in_bytes(copy_range),
        );

        buffer.store(new_buffer.clone());

        new_buffer
    }
}

struct Buffers<T> {
    position: T,
    normal: T,
    uv: T,
}

pub struct VertexBuffers {
    allocator: parking_lot::Mutex<range_alloc::RangeAllocator<u32>>,
    pub position: ArcSwap<wgpu::Buffer>,
    pub normal: ArcSwap<wgpu::Buffer>,
    pub uv: ArcSwap<wgpu::Buffer>,
}

impl VertexBuffers {
    fn size_in_bytes(size: u32, size_of_field: usize) -> u64 {
        size as u64 * size_of_field as u64
    }

    fn create_buffer(
        device: &wgpu::Device,
        label: &str,
        capacity: u32,
        size_of_field: usize,
    ) -> Arc<wgpu::Buffer> {
        Arc::new(device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: Self::size_in_bytes(capacity, size_of_field),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        }))
    }

    pub fn new(capacity: u32, device: &wgpu::Device) -> Self {
        Self {
            allocator: parking_lot::Mutex::new(range_alloc::RangeAllocator::new(0..capacity)),
            position: ArcSwap::from(Self::create_buffer(
                device,
                "position buffer",
                capacity,
                size_of::<Vec3>(),
            )),
            normal: ArcSwap::from(Self::create_buffer(
                device,
                "normal buffer",
                capacity,
                size_of::<Vec3>(),
            )),
            uv: ArcSwap::from(Self::create_buffer(
                device,
                "normal buffer",
                capacity,
                size_of::<Vec2>(),
            )),
        }
    }

    pub(crate) fn insert(
        &self,
        positions: &[Vec3],
        normals: &[Vec3],
        uvs: &[Vec2],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        command_encoder: &mut wgpu::CommandEncoder,
    ) -> Range<u32> {
        let length = positions.len() as u32;

        debug_assert_eq!(positions.len(), normals.len());
        debug_assert_eq!(positions.len(), uvs.len());

        let (buffers, range) = {
            let mut allocator = self.allocator.lock();

            match allocator.allocate_range(length) {
                Ok(range) => {
                    let buffers = Buffers {
                        position: self.position.load_full(),
                        normal: self.normal.load_full(),
                        uv: self.uv.load_full(),
                    };

                    (buffers, range)
                }
                Err(_) => {
                    let new_buffers = Self::resize(
                        &mut allocator,
                        &self.position,
                        &self.normal,
                        &self.uv,
                        length,
                        device,
                        command_encoder,
                    );
                    let range = allocator.allocate_range(length).expect("just resized");
                    (new_buffers, range)
                }
            }
        };

        queue.write_buffer(
            &buffers.position,
            Self::size_in_bytes(range.start, size_of::<Vec3>()),
            bytemuck::cast_slice(positions),
        );

        queue.write_buffer(
            &buffers.normal,
            Self::size_in_bytes(range.start, size_of::<Vec3>()),
            bytemuck::cast_slice(normals),
        );

        queue.write_buffer(
            &buffers.uv,
            Self::size_in_bytes(range.start, size_of::<Vec2>()),
            bytemuck::cast_slice(uvs),
        );

        range
    }

    fn resize(
        allocator: &mut range_alloc::RangeAllocator<u32>,
        position_buffer: &ArcSwap<wgpu::Buffer>,
        normal_buffer: &ArcSwap<wgpu::Buffer>,
        uv_buffer: &ArcSwap<wgpu::Buffer>,
        required_capacity: u32,
        device: &wgpu::Device,
        command_encoder: &mut wgpu::CommandEncoder,
    ) -> Buffers<Arc<wgpu::Buffer>> {
        let copy_range = allocator
            .allocated_ranges()
            .last()
            .map(|range| range.end)
            .unwrap_or(0);

        let old_capacity = allocator.initial_range().end;

        let new_capacity = (old_capacity + required_capacity).max(old_capacity * 2);

        log::info!(
            "Growing vertex buffers from {} to {}",
            old_capacity,
            new_capacity
        );

        allocator.grow_to(new_capacity);

        let new_position_buffer =
            Self::create_buffer(device, "position buffer", new_capacity, size_of::<Vec3>());
        let new_normal_buffer =
            Self::create_buffer(device, "normal buffer", new_capacity, size_of::<Vec3>());
        let new_uv_buffer =
            Self::create_buffer(device, "uv buffer", new_capacity, size_of::<Vec2>());

        command_encoder.copy_buffer_to_buffer(
            &position_buffer.load(),
            0,
            &new_position_buffer,
            0,
            Self::size_in_bytes(copy_range, size_of::<Vec3>()),
        );
        command_encoder.copy_buffer_to_buffer(
            &normal_buffer.load(),
            0,
            &new_normal_buffer,
            0,
            Self::size_in_bytes(copy_range, size_of::<Vec3>()),
        );
        command_encoder.copy_buffer_to_buffer(
            &uv_buffer.load(),
            0,
            &new_uv_buffer,
            0,
            Self::size_in_bytes(copy_range, size_of::<Vec2>()),
        );

        position_buffer.store(new_position_buffer.clone());
        normal_buffer.store(new_normal_buffer.clone());
        uv_buffer.store(new_uv_buffer.clone());

        Buffers {
            position: new_position_buffer,
            normal: new_normal_buffer,
            uv: new_uv_buffer,
        }
    }
}
