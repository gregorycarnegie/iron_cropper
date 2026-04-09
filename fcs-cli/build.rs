fn main() {
    #[cfg(windows)]
    {
        let mut res = winresource::WindowsResource::new();
        res.set_icon("../fcs-gui/assets/app_icon.ico");
        if let Err(err) = res.compile() {
            panic!("failed to compile Windows resources for fcs-cli: {err}");
        }
    }
}
