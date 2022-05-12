cargo build --target wasm32-unknown-unknown --release -Zbuild-std=panic_abort,std &&
wasm-bindgen target/wasm32-unknown-unknown/release/webxr_pbr.wasm --out-dir web/pkg --target web --weak-refs &&
cd web &&
caddy run --config Caddyfile