fn main() {
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../yunet-gui/assets/app_icon.ico");
        if let Err(err) = res.compile() {
            panic!("failed to compile Windows resources for yunet-cli: {err}");
        }
    }
}
