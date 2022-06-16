use wasm_bindgen::closure::Closure;
use wasm_bindgen::JsCast;

pub struct ContextCreationOptions {
    pub stencil: bool,
}

pub struct Canvas {
    pub inner: web_sys::HtmlCanvasElement,
    id: u32,
}

impl Canvas {
    pub fn new_with_id(id: u32) -> Self {
        let canvas: web_sys::HtmlCanvasElement = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .create_element("canvas")
            .unwrap()
            .unchecked_into();

        let body = web_sys::window()
            .unwrap()
            .document()
            .unwrap()
            .body()
            .unwrap();

        canvas
            .set_attribute("data-raw-handle", &id.to_string())
            .unwrap();

        body.append_child(&web_sys::Element::from(canvas.clone()))
            .unwrap();

        Self { inner: canvas, id }
    }

    pub fn create_webgl2_context(
        &self,
        options: ContextCreationOptions,
    ) -> web_sys::WebGl2RenderingContext {
        let mut gl_attribs = std::collections::HashMap::new();
        gl_attribs.insert(String::from("xrCompatible"), true);
        // WebGL silently ignores any stencil writing or testing if this is not set.
        // (Atleast on Chrome). What a fantastic design decision.
        gl_attribs.insert(String::from("stencil"), options.stencil);
        let js_gl_attribs = wasm_bindgen::JsValue::from_serde(&gl_attribs).unwrap();

        self.inner
            .get_context_with_context_options("webgl2", &js_gl_attribs)
            .unwrap()
            .unwrap()
            .dyn_into::<web_sys::WebGl2RenderingContext>()
            .unwrap()
    }
}

impl Default for Canvas {
    fn default() -> Self {
        Self::new_with_id(0)
    }
}

unsafe impl raw_window_handle::HasRawWindowHandle for Canvas {
    fn raw_window_handle(&self) -> raw_window_handle::RawWindowHandle {
        let mut web = raw_window_handle::WebHandle::empty();
        web.id = self.id;

        raw_window_handle::RawWindowHandle::Web(web)
    }
}

pub(crate) fn request_animation_frame(
    session: &web_sys::XrSession,
    f: &Closure<dyn FnMut(f64, web_sys::XrFrame)>,
) -> u32 {
    // This turns the Closure into a js_sys::Function
    // See https://rustwasm.github.io/wasm-bindgen/api/wasm_bindgen/closure/struct.Closure.html#casting-a-closure-to-a-js_sysfunction
    session.request_animation_frame(f.as_ref().unchecked_ref())
}

pub struct Session {
    pub inner: web_sys::XrSession,
}

impl Session {
    pub fn run_rendering_loop<F: FnMut(f64, web_sys::XrFrame) + 'static>(&self, mut func: F) {
        use std::cell::RefCell;
        use std::rc::Rc;

        // Wierd hacky closure stuff that I don't understand. Taken from wasm-bindgen.
        // TODO: link source.
        let closure = Rc::new(RefCell::new(None));
        let closure_clone = closure.clone();

        *closure.borrow_mut() = Some(Closure::wrap(Box::new(
            move |time: f64, frame: web_sys::XrFrame| {
                let session = frame.session();

                request_animation_frame(&session, closure_clone.borrow().as_ref().unwrap());

                func(time, frame);
            },
        )
            as Box<dyn FnMut(f64, web_sys::XrFrame)>));

        request_animation_frame(&self.inner, closure.borrow().as_ref().unwrap());
    }
}

pub fn create_button(text: &str) -> web_sys::HtmlButtonElement {
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

pub async fn button_click_future(button: &web_sys::HtmlButtonElement) {
    wasm_bindgen_futures::JsFuture::from(js_sys::Promise::new(&mut |resolve, _reject| {
        button.set_onclick(Some(&resolve))
    }))
    .await
    .unwrap();
}
pub fn append_break() {
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
