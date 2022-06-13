use std::mem::size_of;
use std::ops::Range;

use super::Instance;
use glam::{Vec2, Vec3};

pub(crate) type InstanceBuffer = VecGpuBuffer<Instance>;

pub(crate) struct VecGpuBuffer<T: bytemuck::Pod> {
    offset: u32,
    capacity: u32,
    pub(crate) buffer: wgpu::Buffer,
    usage: wgpu::BufferUsages,
    _phantom: std::marker::PhantomData<T>,
    label: &'static str,
}

impl<T: bytemuck::Pod> VecGpuBuffer<T> {
    fn size_in_bytes(size: u32) -> u64 {
        size as u64 * size_of::<T>() as u64
    }

    pub(crate) fn new(
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

    pub(crate) fn clear(&mut self) {
        self.offset = 0;
    }

    pub(crate) fn push(
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

#[allow(dead_code)]
pub(crate) struct IndexBuffer {
    allocator: range_alloc::RangeAllocator<u32>,
    pub(crate) buffer: wgpu::Buffer,
}

#[allow(dead_code)]
impl IndexBuffer {
    fn size_in_bytes(size: u32) -> u64 {
        size as u64 * size_of::<u32>() as u64
    }

    pub(crate) fn new(capacity: u32, device: &wgpu::Device) -> Self {
        Self {
            allocator: range_alloc::RangeAllocator::new(0..capacity),
            buffer: device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("index buffer"),
                size: Self::size_in_bytes(capacity),
                usage: wgpu::BufferUsages::INDEX
                    | wgpu::BufferUsages::COPY_DST
                    | wgpu::BufferUsages::COPY_SRC,
                mapped_at_creation: false,
            }),
        }
    }

    fn insert(
        &mut self,
        indices: &[u32],
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        command_encoder: &mut wgpu::CommandEncoder,
    ) -> Range<u32> {
        let length = indices.len() as u32;

        let range = match self.allocator.allocate_range(length) {
            Ok(range) => range,
            Err(_) => {
                self.resize(length, device, command_encoder);
                self.allocator.allocate_range(length).expect("just resized")
            }
        };

        queue.write_buffer(
            &self.buffer,
            Self::size_in_bytes(range.start),
            bytemuck::cast_slice(indices),
        );

        range
    }

    fn resize(
        &mut self,
        required_capacity: u32,
        device: &wgpu::Device,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        let copy_range = self
            .allocator
            .allocated_ranges()
            .last()
            .map(|range| range.end)
            .unwrap_or(0);

        let old_capacity = self.allocator.initial_range().end;

        let new_capacity = (old_capacity + required_capacity).max(old_capacity * 2);

        log::info!(
            "Growing index buffer from {} to {}",
            old_capacity,
            new_capacity
        );

        self.allocator.grow_to(new_capacity);

        let new_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("index buffer"),
            size: Self::size_in_bytes(new_capacity),
            usage: wgpu::BufferUsages::INDEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        command_encoder.copy_buffer_to_buffer(
            &self.buffer,
            0,
            &new_buffer,
            0,
            Self::size_in_bytes(copy_range),
        );

        self.buffer = new_buffer;
    }
}

pub(crate) struct VertexBuffers {
    allocator: range_alloc::RangeAllocator<u32>,
    pub(crate) position: wgpu::Buffer,
    pub(crate) normal: wgpu::Buffer,
    pub(crate) uv: wgpu::Buffer,
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
    ) -> wgpu::Buffer {
        device.create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: Self::size_in_bytes(capacity, size_of_field),
            usage: wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        })
    }

    pub(crate) fn new(capacity: u32, device: &wgpu::Device) -> Self {
        Self {
            allocator: range_alloc::RangeAllocator::new(0..capacity),
            position: Self::create_buffer(device, "position buffer", capacity, size_of::<Vec3>()),
            normal: Self::create_buffer(device, "normal buffer", capacity, size_of::<Vec3>()),
            uv: Self::create_buffer(device, "normal buffer", capacity, size_of::<Vec2>()),
        }
    }

    pub(crate) fn insert(
        &mut self,
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

        let range = match self.allocator.allocate_range(length) {
            Ok(range) => range,
            Err(_) => {
                self.resize(length, device, command_encoder);
                self.allocator.allocate_range(length).expect("just resized")
            }
        };

        queue.write_buffer(
            &self.position,
            Self::size_in_bytes(range.start, size_of::<Vec3>()),
            bytemuck::cast_slice(positions),
        );

        queue.write_buffer(
            &self.normal,
            Self::size_in_bytes(range.start, size_of::<Vec3>()),
            bytemuck::cast_slice(normals),
        );

        queue.write_buffer(
            &self.uv,
            Self::size_in_bytes(range.start, size_of::<Vec2>()),
            bytemuck::cast_slice(uvs),
        );

        range
    }

    fn resize(
        &mut self,
        required_capacity: u32,
        device: &wgpu::Device,
        command_encoder: &mut wgpu::CommandEncoder,
    ) {
        let copy_range = self
            .allocator
            .allocated_ranges()
            .last()
            .map(|range| range.end)
            .unwrap_or(0);

        let old_capacity = self.allocator.initial_range().end;

        let new_capacity = (old_capacity + required_capacity).max(old_capacity * 2);

        log::info!(
            "Growing vertex buffers from {} to {}",
            old_capacity,
            new_capacity
        );

        self.allocator.grow_to(new_capacity);

        let new_position_buffer =
            Self::create_buffer(device, "position buffer", new_capacity, size_of::<Vec3>());
        let new_normal_buffer =
            Self::create_buffer(device, "normal buffer", new_capacity, size_of::<Vec3>());
        let new_uv_buffer =
            Self::create_buffer(device, "uv buffer", new_capacity, size_of::<Vec2>());

        command_encoder.copy_buffer_to_buffer(
            &self.position,
            0,
            &new_position_buffer,
            0,
            Self::size_in_bytes(copy_range, size_of::<Vec3>()),
        );
        command_encoder.copy_buffer_to_buffer(
            &self.normal,
            0,
            &new_normal_buffer,
            0,
            Self::size_in_bytes(copy_range, size_of::<Vec3>()),
        );
        command_encoder.copy_buffer_to_buffer(
            &self.uv,
            0,
            &new_uv_buffer,
            0,
            Self::size_in_bytes(copy_range, size_of::<Vec2>()),
        );

        self.position = new_position_buffer;
        self.normal = new_normal_buffer;
        self.uv = new_uv_buffer;
    }
}
