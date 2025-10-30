# Test Fixtures

Store sample images, golden JSON outputs, and reference metadata for parity testing in this directory. Individual fixture assets should not be committed if they contain proprietary or sensitive information; prefer synthetic or cleared data.

Recommended layout:

- `fixtures/images/` - raw input frames used across unit and integration tests.
- `fixtures/golden/` - serialized detection outputs for regression comparisons.
- `fixtures/opencv/` - OpenCV YuNet reference outputs captured for parity validation.

Remember to update `.gitignore` rules when adding new tracked files here.
