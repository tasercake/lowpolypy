use leptos::html::Canvas;
use leptos::prelude::*;
use web_sys::wasm_bindgen::JsCast;
use web_sys::CanvasRenderingContext2d;

fn main() {
    leptos::mount::mount_to_body(|| {
        view! {
            <DrawingComponent />
        }
    })
}

#[component]
fn DrawingComponent() -> impl IntoView {
    let canvas_ref = NodeRef::<Canvas>::new();

    Effect::new(move |_| {
        draw_face(canvas_ref);
    });

    view! {
        <canvas
            width="400"
            height="400"
            style="border: 1px solid black"
            node_ref=canvas_ref
        />
    }
}

fn draw_face(canvas_ref: NodeRef<Canvas>) {
    // let canvas: HtmlCanvasElement = canvas.get::<HtmlCanvasElement>().unwrap();

    // Get the canvas element
    let canvas: web_sys::HtmlCanvasElement = canvas_ref.get().unwrap();

    let uncanvas = canvas.get_context("2d").unwrap().unwrap();

    let context: CanvasRenderingContext2d = uncanvas.dyn_into().unwrap();

    // Draw face circle
    context.begin_path();
    context
        .arc(200.0, 200.0, 150.0, 0.0, 2.0 * std::f64::consts::PI)
        .unwrap();
    // context.set_stroke_style(&"black".into());
    context.stroke();

    // Draw eyes
    context.begin_path();
    context
        .arc(140.0, 150.0, 20.0, 0.0, 2.0 * std::f64::consts::PI)
        .unwrap();
    context
        .arc(260.0, 150.0, 20.0, 0.0, 2.0 * std::f64::consts::PI)
        .unwrap();
    // context.set_fill_style(&"black".into());
    context.fill();

    // Draw smile
    context.begin_path();
    context
        .arc(200.0, 200.0, 80.0, 0.0, std::f64::consts::PI)
        .unwrap();
    context.stroke();
}
