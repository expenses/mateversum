pub mod models;
pub mod textures;

use std::ops::Range;

pub trait HttpClient {
    type Future: std::future::Future<Output = anyhow::Result<Vec<u8>>>;

    fn fetch_bytes(&self, url: &url::Url, range: Option<Range<usize>>) -> Self::Future;
}