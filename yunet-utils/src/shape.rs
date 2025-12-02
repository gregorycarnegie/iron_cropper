//! Shape utilities for custom crop geometry and masking.
//!
//! Provides the `CropShape` enum shared across the workspace together with helpers
//! for generating polygon outlines and applying alpha masks to RGBA images.

use image::{DynamicImage, RgbaImage};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Transform};

/// Polygon corner styles.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "style", rename_all = "snake_case")]
pub enum PolygonCornerStyle {
    Sharp,
    Rounded { radius_pct: f32 },
    Chamfered { size_pct: f32 },
    Bezier { tension: f32 },
}

impl Default for PolygonCornerStyle {
    fn default() -> Self {
        Self::Sharp
    }
}

/// Shapes supported by the crop exporter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CropShape {
    Rectangle,
    RoundedRectangle {
        radius_pct: f32,
    },
    ChamferedRectangle {
        size_pct: f32,
    },
    Ellipse,
    Polygon {
        sides: u8,
        rotation_deg: f32,
        #[serde(default)]
        corner_style: PolygonCornerStyle,
    },
    Star {
        points: u8,
        inner_radius_pct: f32,
        rotation_deg: f32,
    },
}

impl CropShape {
    /// Sanitize values to keep them in a sensible range.
    pub fn sanitized(&self) -> Self {
        match self {
            Self::Rectangle => Self::Rectangle,
            Self::Ellipse => Self::Ellipse,
            Self::RoundedRectangle { radius_pct } => Self::RoundedRectangle {
                radius_pct: radius_pct.clamp(0.0, 0.5),
            },
            Self::ChamferedRectangle { size_pct } => Self::ChamferedRectangle {
                size_pct: size_pct.clamp(0.0, 0.5),
            },
            Self::Polygon {
                sides,
                rotation_deg,
                corner_style,
            } => {
                let sides = (*sides).clamp(3u8, 24u8);
                let rotation_deg = normalize_angle(*rotation_deg);
                let corner_style = match corner_style {
                    PolygonCornerStyle::Sharp => PolygonCornerStyle::Sharp,
                    PolygonCornerStyle::Rounded { radius_pct } => PolygonCornerStyle::Rounded {
                        radius_pct: radius_pct.clamp(0.0, 0.5),
                    },
                    PolygonCornerStyle::Chamfered { size_pct } => PolygonCornerStyle::Chamfered {
                        size_pct: size_pct.clamp(0.0, 0.5),
                    },
                    PolygonCornerStyle::Bezier { tension } => PolygonCornerStyle::Bezier {
                        tension: tension.clamp(0.0, 2.0),
                    },
                };
                Self::Polygon {
                    sides,
                    rotation_deg,
                    corner_style,
                }
            }
            Self::Star {
                points,
                inner_radius_pct,
                rotation_deg,
            } => {
                let points = (*points).clamp(3u8, 24u8);
                let inner_radius_pct = inner_radius_pct.clamp(0.1, 0.9);
                let rotation_deg = normalize_angle(*rotation_deg);
                Self::Star {
                    points,
                    inner_radius_pct,
                    rotation_deg,
                }
            }
        }
    }
}

impl Default for CropShape {
    fn default() -> Self {
        Self::Rectangle
    }
}

/// Single 2D point.
#[derive(Debug, Clone, Copy)]
struct Point {
    x: f32,
    y: f32,
}

fn normalize_angle(mut angle: f32) -> f32 {
    while angle < 0.0 {
        angle += 360.0;
    }
    while angle >= 360.0 {
        angle -= 360.0;
    }
    angle
}

/// Generate outline points for a shape fitted to the supplied width/height.
fn outline_points(width: u32, height: u32, shape: &CropShape) -> Vec<Point> {
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    let shape = shape.sanitized();

    match shape {
        CropShape::Rectangle => vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: w, y: 0.0 },
            Point { x: w, y: h },
            Point { x: 0.0, y: h },
        ],
        CropShape::Ellipse => {
            let cx = w / 2.0;
            let cy = h / 2.0;
            let rx = w / 2.0;
            let ry = h / 2.0;
            let segments = 128;
            (0..segments)
                .map(|i| {
                    let theta = (i as f32 / segments as f32) * 2.0 * PI;
                    Point {
                        x: cx + rx * theta.cos(),
                        y: cy + ry * theta.sin(),
                    }
                })
                .collect()
        }
        CropShape::RoundedRectangle { radius_pct } => {
            let radius = (w.min(h) * radius_pct).clamp(0.0, w.min(h) / 2.0);
            rounded_rect_points(w, h, radius, 16)
        }
        CropShape::ChamferedRectangle { size_pct } => {
            let inset = (w.min(h) * size_pct).clamp(0.0, w.min(h) / 2.0);
            chamfered_rect_points(w, h, inset)
        }
        CropShape::Polygon {
            sides,
            rotation_deg,
            corner_style,
        } => polygon_points(w, h, sides, rotation_deg, corner_style),
        CropShape::Star {
            points,
            inner_radius_pct,
            rotation_deg,
        } => star_points(w, h, points, inner_radius_pct, rotation_deg),
    }
}

fn rounded_rect_points(width: f32, height: f32, radius: f32, segments: usize) -> Vec<Point> {
    if radius <= 0.0 {
        return vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: width, y: 0.0 },
            Point {
                x: width,
                y: height,
            },
            Point { x: 0.0, y: height },
        ];
    }

    let mut points = Vec::with_capacity(segments * 4);
    let mut add_corner = |cx: f32, cy: f32, start: f32, end: f32| {
        let steps = segments.max(3);
        let delta = (end - start) / steps as f32;
        for i in 0..=steps {
            let angle = start + delta * i as f32;
            points.push(Point {
                x: cx + radius * angle.cos(),
                y: cy + radius * angle.sin(),
            });
        }
    };

    // Top-right corner (angles -90 -> 0 degrees)
    add_corner(width - radius, radius, -PI / 2.0, 0.0);
    // Bottom-right (0 -> 90)
    add_corner(width - radius, height - radius, 0.0, PI / 2.0);
    // Bottom-left (90 -> 180)
    add_corner(radius, height - radius, PI / 2.0, PI);
    // Top-left (180 -> 270)
    add_corner(radius, radius, PI, 1.5 * PI);

    points
}

fn chamfered_rect_points(width: f32, height: f32, inset: f32) -> Vec<Point> {
    if inset <= 0.0 {
        return vec![
            Point { x: 0.0, y: 0.0 },
            Point { x: width, y: 0.0 },
            Point {
                x: width,
                y: height,
            },
            Point { x: 0.0, y: height },
        ];
    }

    vec![
        Point { x: inset, y: 0.0 },
        Point {
            x: width - inset,
            y: 0.0,
        },
        Point { x: width, y: inset },
        Point {
            x: width,
            y: height - inset,
        },
        Point {
            x: width - inset,
            y: height,
        },
        Point {
            x: inset,
            y: height,
        },
        Point {
            x: 0.0,
            y: height - inset,
        },
        Point { x: 0.0, y: inset },
    ]
}

fn polygon_points(
    width: f32,
    height: f32,
    sides: u8,
    rotation_deg: f32,
    corner_style: PolygonCornerStyle,
) -> Vec<Point> {
    let n = sides.max(3) as usize;
    let cx = width / 2.0;
    let cy = height / 2.0;
    let radius = 0.5 * width.min(height);
    let rotation = rotation_deg.to_radians();

    let mut base_vertices = Vec::with_capacity(n);
    for i in 0..n {
        let angle = rotation + 2.0 * PI * i as f32 / n as f32;
        base_vertices.push(Point {
            x: cx + radius * angle.cos(),
            y: cy + radius * angle.sin(),
        });
    }

    match corner_style {
        PolygonCornerStyle::Sharp => base_vertices,
        PolygonCornerStyle::Chamfered { size_pct } => {
            let inset = (width.min(height) * size_pct).clamp(0.0, radius);
            chamfer_polygon(&base_vertices, inset)
        }
        PolygonCornerStyle::Rounded { radius_pct } => {
            let r = (width.min(height) * radius_pct).clamp(0.0, radius);
            rounded_polygon(&base_vertices, r, 8)
        }
        PolygonCornerStyle::Bezier { tension } => bezier_polygon(&base_vertices, tension, 16),
    }
}

fn chamfer_polygon(vertices: &[Point], inset: f32) -> Vec<Point> {
    if inset <= 0.0 {
        return vertices.to_vec();
    }

    let len = vertices.len();
    let mut points = Vec::with_capacity(len * 2);

    for i in 0..len {
        let prev = vertices[(i + len - 1) % len];
        let current = vertices[i];
        let next = vertices[(i + 1) % len];

        let prev_vec = normalize(Point {
            x: current.x - prev.x,
            y: current.y - prev.y,
        });
        let next_vec = normalize(Point {
            x: next.x - current.x,
            y: next.y - current.y,
        });

        let prev_edge_len = distance(prev, current);
        let next_edge_len = distance(current, next);
        let offset_prev = inset.min(prev_edge_len / 2.0);
        let offset_next = inset.min(next_edge_len / 2.0);

        points.push(Point {
            x: current.x - prev_vec.x * offset_prev,
            y: current.y - prev_vec.y * offset_prev,
        });
        points.push(Point {
            x: current.x + next_vec.x * offset_next,
            y: current.y + next_vec.y * offset_next,
        });
    }

    points
}

fn rounded_polygon(vertices: &[Point], radius: f32, segments: usize) -> Vec<Point> {
    if radius <= 0.0 {
        return vertices.to_vec();
    }

    let len = vertices.len();
    let mut points = Vec::with_capacity(len * segments);

    for i in 0..len {
        let prev = vertices[(i + len - 1) % len];
        let current = vertices[i];
        let next = vertices[(i + 1) % len];

        let incoming = normalize(Point {
            x: current.x - prev.x,
            y: current.y - prev.y,
        });
        let outgoing = normalize(Point {
            x: next.x - current.x,
            y: next.y - current.y,
        });

        let angle_cos = (-incoming.x * outgoing.x) + (-incoming.y * outgoing.y);
        let angle_cos = angle_cos.clamp(-0.999_9, 0.999_9);
        let half_angle = 0.5 * angle_cos.acos();
        let mut offset = radius / half_angle.tan();
        let incoming_len = distance(prev, current);
        let outgoing_len = distance(current, next);
        offset = offset.min(incoming_len / 2.0).min(outgoing_len / 2.0);

        let start = Point {
            x: current.x - incoming.x * offset,
            y: current.y - incoming.y * offset,
        };
        let end = Point {
            x: current.x + outgoing.x * offset,
            y: current.y + outgoing.y * offset,
        };

        let bisector = normalize(Point {
            x: -incoming.x + outgoing.x,
            y: -incoming.y + outgoing.y,
        });
        let center_distance = radius / half_angle.sin();
        let center = Point {
            x: current.x + bisector.x * center_distance,
            y: current.y + bisector.y * center_distance,
        };

        let start_angle = (start.y - center.y).atan2(start.x - center.x);
        let end_angle = (end.y - center.y).atan2(end.x - center.x);
        let mut delta = end_angle - start_angle;
        while delta <= 0.0 {
            delta += 2.0 * PI;
        }
        let steps = segments.max(3);
        let step = delta / steps as f32;
        for j in 0..=steps {
            let angle = start_angle + step * j as f32;
            points.push(Point {
                x: center.x + radius * angle.cos(),
                y: center.y + radius * angle.sin(),
            });
        }
    }

    points
}

fn bezier_polygon(vertices: &[Point], tension: f32, segments: usize) -> Vec<Point> {
    if tension <= 0.0 {
        return vertices.to_vec();
    }

    let len = vertices.len();
    let mut points = Vec::with_capacity(len * segments);

    // Calculate control points for each vertex
    // We use a simple cardinal spline approach where the control points are derived
    // from the previous and next vertices.
    let mut control_points = Vec::with_capacity(len * 2);

    for i in 0..len {
        let prev = vertices[(i + len - 1) % len];
        let current = vertices[i];
        let next = vertices[(i + 1) % len];

        // Tangent vector at current point
        let tangent_x = next.x - prev.x;
        let tangent_y = next.y - prev.y;

        // Scale by tension
        // tension of 0.5 is standard Catmull-Rom
        let cp_dist = tension * 0.5; // Adjust scaling factor as needed

        // Control point "before" current (incoming)
        let cp1 = Point {
            x: current.x - tangent_x * cp_dist,
            y: current.y - tangent_y * cp_dist,
        };
        // Control point "after" current (outgoing)
        let cp2 = Point {
            x: current.x + tangent_x * cp_dist,
            y: current.y + tangent_y * cp_dist,
        };

        control_points.push((cp1, cp2));
    }

    // Generate curve segments between vertices
    for i in 0..len {
        let p0 = vertices[i];
        let p1 = vertices[(i + 1) % len];

        // Control points for this segment:
        // p0 -> cp_after_p0 -> cp_before_p1 -> p1
        let cp1 = control_points[i].1;
        let cp2 = control_points[(i + 1) % len].0;

        for j in 0..segments {
            let t = j as f32 / segments as f32;
            points.push(cubic_bezier(p0, cp1, cp2, p1, t));
        }
    }

    points
}

fn cubic_bezier(p0: Point, p1: Point, p2: Point, p3: Point, t: f32) -> Point {
    let t2 = t * t;
    let t3 = t2 * t;
    let mt = 1.0 - t;
    let mt2 = mt * mt;
    let mt3 = mt2 * mt;

    Point {
        x: mt3 * p0.x + 3.0 * mt2 * t * p1.x + 3.0 * mt * t2 * p2.x + t3 * p3.x,
        y: mt3 * p0.y + 3.0 * mt2 * t * p1.y + 3.0 * mt * t2 * p2.y + t3 * p3.y,
    }
}

fn distance(a: Point, b: Point) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

fn normalize(v: Point) -> Point {
    let len = (v.x * v.x + v.y * v.y).sqrt();
    if len <= f32::EPSILON {
        Point { x: 0.0, y: 0.0 }
    } else {
        Point {
            x: v.x / len,
            y: v.y / len,
        }
    }
}

fn build_path(width: u32, height: u32, shape: &CropShape) -> Option<tiny_skia::Path> {
    let points = outline_points(width, height, shape);
    if points.is_empty() {
        return None;
    }

    let mut builder = PathBuilder::new();
    builder.move_to(points[0].x, points[0].y);
    for point in points.iter().skip(1) {
        builder.line_to(point.x, point.y);
    }
    builder.close();
    builder.finish()
}

/// Apply the shape mask to the supplied RGBA image in-place.
pub fn apply_shape_mask(image: &mut RgbaImage, shape: &CropShape) {
    if matches!(shape, CropShape::Rectangle) {
        return;
    }

    let width = image.width();
    let height = image.height();

    let mut pixmap = match Pixmap::new(width, height) {
        Some(p) => p,
        None => return,
    };
    pixmap.fill(tiny_skia::Color::from_rgba8(0, 0, 0, 0));

    if let Some(path) = build_path(width, height, shape) {
        let mut paint = Paint::default();
        paint.set_color_rgba8(255, 255, 255, 255);
        pixmap.fill_path(
            &path,
            &paint,
            FillRule::Winding,
            Transform::identity(),
            None,
        );

        let mask = pixmap.data();
        for (pixel, mask_px) in image.pixels_mut().zip(mask.chunks_exact(4)) {
            let alpha = mask_px[3];
            if alpha == 0 {
                pixel[0] = 0;
                pixel[1] = 0;
                pixel[2] = 0;
            }
            pixel[3] = alpha;
        }
    }
}

fn star_points(
    width: f32,
    height: f32,
    points: u8,
    inner_radius_pct: f32,
    rotation_deg: f32,
) -> Vec<Point> {
    let n = points.max(3) as usize;
    let cx = width / 2.0;
    let cy = height / 2.0;
    let outer_radius = 0.5 * width.min(height);
    let inner_radius = outer_radius * inner_radius_pct;
    let rotation = rotation_deg.to_radians();

    let mut vertices = Vec::with_capacity(n * 2);
    let step_angle = PI / n as f32;

    for i in 0..n {
        // Outer point
        let angle_outer = rotation + 2.0 * PI * i as f32 / n as f32;
        vertices.push(Point {
            x: cx + outer_radius * angle_outer.cos(),
            y: cy + outer_radius * angle_outer.sin(),
        });

        // Inner point
        let angle_inner = angle_outer + step_angle;
        vertices.push(Point {
            x: cx + inner_radius * angle_inner.cos(),
            y: cy + inner_radius * angle_inner.sin(),
        });
    }

    vertices
}

/// Apply the shape mask to a dynamic image, upgrading to RGBA as needed.
pub fn apply_shape_mask_dynamic(image: &mut DynamicImage, shape: &CropShape) {
    if matches!(shape, CropShape::Rectangle) {
        return;
    }

    let mut rgba = image.to_rgba8();
    apply_shape_mask(&mut rgba, shape);
    *image = DynamicImage::ImageRgba8(rgba);
}

/// Generate outline points scaled to an arbitrary rectangle.
pub fn outline_points_for_rect(
    rect_width: f32,
    rect_height: f32,
    shape: &CropShape,
) -> Vec<(f32, f32)> {
    let width_px = rect_width.max(1.0).round() as u32;
    let height_px = rect_height.max(1.0).round() as u32;
    outline_points(width_px, height_px, shape)
        .into_iter()
        .map(|p| (p.x, p.y))
        .collect()
}
