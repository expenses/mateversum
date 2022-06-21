use std::ops::Range;

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
    pub(crate) fn new(cache: web_sys::Cache) -> anyhow::Result<Self> {
        Ok(Self {
            cache,
            ipfs_gateway_url: url::Url::parse("http://localhost:8080/ipfs")?,
        })
    }

    pub(crate) async fn fetch_uint8_array(
        &self,
        url: &url::Url,
        byte_range: Option<Range<usize>>,
        cache: bool,
    ) -> anyhow::Result<js_sys::Uint8Array> {
        let request_init = construct_request_init(byte_range.clone())?;

        let mut cache_url = url.clone();
        let mut fetch_url = url.clone();

        if url.scheme() == "ipfs" {
            fetch_url = self.ipfs_gateway_url.clone();

            let host_err = || anyhow::anyhow!("Failed to get url host");
            let path_segments_err = || anyhow::anyhow!("Failed to get url path segments");

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

            cache_url = url::Url::parse("http://ipfs").unwrap();

            // Append the host_str (the CID in this case) to the path.
            let new_path: Vec<_> = std::iter::once(url.host_str().ok_or_else(host_err)?)
                .chain(url.path_segments().into_iter().flatten())
                .map(|string| string.to_owned())
                .collect();
            cache_url
                .path_segments_mut()
                .map_err(|_| path_segments_err())?
                .extend(new_path);
        }

        if let Some(byte_range) = byte_range.clone() {
            cache_url.query_pairs_mut().append_pair(
                "bytes",
                &format!("{}-{}", byte_range.start, byte_range.end - 1),
            );
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

    pub(crate) async fn fetch_bytes(
        &self,
        url: &url::Url,
        byte_range: Option<Range<usize>>,
    ) -> anyhow::Result<Vec<u8>> {
        let uint8_array = self.fetch_uint8_array(url, byte_range, true).await?;

        Ok(uint8_array.to_vec())
    }

    pub(crate) async fn fetch_bytes_without_caching(
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
