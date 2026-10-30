//! The web UI, compiled in, and how it becomes one self-contained HTML file.
//!
//! `web/index.html` links its stylesheet and scripts by fixed tags; they're replaced here by the
//! files' contents, and the data slot by `window.PUGGLES_DATA = ...`. With no data the page runs
//! in server mode and talks to the [`Dashboard`](super::Dashboard) API instead.

use std::io;
use std::process::{Command, Stdio};

const INDEX: &str = include_str!("web/index.html");
const ASSETS: [(&str, &str, &str, &str); 3] = [
    ("<link rel=\"stylesheet\" href=\"static/style.css\">", include_str!("web/style.css"), "<style>\n", "\n</style>"),
    ("<script src=\"static/plot.js\"></script>", include_str!("web/plot.js"), "<script>\n", "\n</script>"),
    ("<script src=\"static/app.js\"></script>", include_str!("web/app.js"), "<script>\n", "\n</script>"),
];
const DATA_SLOT: &str = "<script>/*PUGGLES_DATA*/</script>";

/// The UI with `data_json` (a JSON object: `{"kind": "run", ...}` or `{"kind": "plot", ...}`)
/// embedded, as one HTML file that opens anywhere without a server.
pub fn standalone_html(data_json: &str) -> String {
    // `</` inside a <script> would end it early; `<\/` is the same string to JavaScript.
    let payload = format!("<script>window.PUGGLES_DATA = {};</script>", data_json.replace("</", "<\\/"));
    inline_assets(&INDEX.replace(DATA_SLOT, &payload))
}

/// The UI in server mode (no embedded data).
pub(crate) fn server_page() -> String {
    inline_assets(INDEX)
}

fn inline_assets(html: &str) -> String {
    let mut out = html.to_string();
    for (tag, content, open, close) in ASSETS {
        debug_assert!(out.contains(tag), "index.html is missing {tag}");
        out = out.replace(tag, &format!("{open}{content}{close}"));
    }
    out
}

/// Open a URL or file in the default browser, without waiting for it. Does nothing when the
/// `PUGGLES_NO_BROWSER` environment variable is set (CI, headless machines).
pub fn open_in_browser(target: &str) -> io::Result<()> {
    if std::env::var_os("PUGGLES_NO_BROWSER").is_some() {
        return Ok(());
    }
    #[cfg(target_os = "macos")]
    let mut cmd = Command::new("open");
    #[cfg(target_os = "windows")]
    let mut cmd = {
        let mut c = Command::new("rundll32");
        c.arg("url.dll,FileProtocolHandler");
        c
    };
    #[cfg(not(any(target_os = "macos", target_os = "windows")))]
    let mut cmd = Command::new("xdg-open");
    let mut child = cmd.arg(target).stdin(Stdio::null()).stdout(Stdio::null()).stderr(Stdio::null()).spawn()?;
    // Reap the launcher so it doesn't linger as a zombie.
    std::thread::spawn(move || {
        let _ = child.wait();
    });
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn assets_are_inlined_and_data_is_escaped() {
        let page = standalone_html(r#"{"kind":"plot","spec":{"labs":{"title":"</script><b>"}}}"#);
        for (tag, ..) in ASSETS {
            assert!(!page.contains(tag), "{tag} should have been inlined");
        }
        assert!(page.contains("window.PUGGLES_DATA = "));
        assert!(page.contains(r#"<\/script><b>"#), "embedded data must not close the script tag");
        assert!(!page.contains(DATA_SLOT));
        assert!(page.contains("window.PugglesPlot"), "plot.js inlined");

        let server = server_page();
        assert!(!server.contains("window.PUGGLES_DATA = "), "server mode has no embedded data");
    }

    #[test]
    fn scripts_never_contain_a_closing_script_tag() {
        // An inlined script containing `</script>` would cut the page short.
        for (_, content, ..) in &ASSETS[1..] {
            assert!(!content.contains("</script"));
        }
    }
}
