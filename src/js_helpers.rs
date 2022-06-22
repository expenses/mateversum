use wasm_bindgen::JsCast;

pub(crate) fn create_button(text: &str) -> web_sys::HtmlButtonElement {
    let button: web_sys::HtmlButtonElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("button")
        .unwrap()
        .unchecked_into();

    button.set_inner_text(text);

    let body = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .body()
        .unwrap();

    body.append_child(&web_sys::Element::from(button.clone()))
        .unwrap();

    button
}

pub(crate) async fn button_click_future(button: &web_sys::HtmlButtonElement) {
    wasm_bindgen_futures::JsFuture::from(js_sys::Promise::new(&mut |resolve, _reject| {
        button.set_onclick(Some(&resolve))
    }))
    .await
    .unwrap();
}
pub(crate) fn append_break() {
    let br: web_sys::HtmlBrElement = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .create_element("br")
        .unwrap()
        .unchecked_into();

    let body = web_sys::window()
        .unwrap()
        .document()
        .unwrap()
        .body()
        .unwrap();

    body.append_child(&web_sys::Element::from(br)).unwrap();
}
