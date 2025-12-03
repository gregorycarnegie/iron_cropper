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
    KochPolygon {
        sides: u8,
        rotation_deg: f32,
        iterations: u8,
    },
    KochRectangle {
        iterations: u8,
    },
}

impl CropShape {
    /// Sanitize values to keep them in a sensible range.
    pub fn sanitized(&self) -> Self {
        match self {
            CropShape::Rectangle => CropShape::Rectangle,
            CropShape::RoundedRectangle { radius_pct } => CropShape::RoundedRectangle {
                radius_pct: radius_pct.clamp(0.0, 0.5),
            },
            CropShape::ChamferedRectangle { size_pct } => CropShape::ChamferedRectangle {
                size_pct: size_pct.clamp(0.0, 0.5),
            },
            CropShape::Ellipse => Self::Ellipse,
            CropShape::Polygon {
                sides,
                rotation_deg,
                corner_style,
            } => CropShape::Polygon {
                sides: (*sides).max(3),
                rotation_deg: *rotation_deg,
                corner_style: match corner_style {
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
                },
            },
            CropShape::Star {
                points,
                inner_radius_pct,
                rotation_deg,
            } => CropShape::Star {
                points: (*points).max(3),
                inner_radius_pct: inner_radius_pct.clamp(0.0, 1.0),
                rotation_deg: *rotation_deg,
            },
            CropShape::KochPolygon {
                sides,
                rotation_deg,
                iterations,
            } => CropShape::KochPolygon {
                sides: (*sides).max(3),
                rotation_deg: *rotation_deg,
                iterations: (*iterations).min(5),
            },
            CropShape::KochRectangle { iterations } => CropShape::KochRectangle {
                iterations: (*iterations).min(5),
            },
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

/// Generate outline points for a shape fitted to the supplied width/height.
fn outline_points(width: u32, height: u32, shape: &CropShape) -> Vec<Point> {
    let w = width.max(1) as f32;
    let h = height.max(1) as f32;
    let shape = shape.sanitized();

    let mut points = match &shape {
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
        } => polygon_points(w, h, *sides, *rotation_deg, corner_style.clone()),
        CropShape::Star {
            points,
            inner_radius_pct,
            rotation_deg,
        } => star_points(w, h, *points, *inner_radius_pct, *rotation_deg),
        CropShape::KochPolygon {
            sides,
            rotation_deg,
            iterations,
        } => {
            let base_poly = polygon_points(w, h, *sides, *rotation_deg, PolygonCornerStyle::Sharp);
            koch_fractal(&base_poly, *iterations)
        }
        CropShape::KochRectangle { iterations } => {
            let base_rect = vec![
                Point { x: 0.0, y: 0.0 },
                Point { x: w, y: 0.0 },
                Point { x: w, y: h },
                Point { x: 0.0, y: h },
            ];
            koch_fractal(&base_rect, *iterations)
        }
    };

    // Fit complex shapes to bounds to prevent clipping
    match shape {
        CropShape::Polygon { .. }
        | CropShape::Star { .. }
        | CropShape::KochPolygon { .. }
        | CropShape::KochRectangle { .. } => {
            fit_points_to_bounds(&mut points, w, h);
        }
        _ => {}
    }

    points
}

fn fit_points_to_bounds(points: &mut [Point], width: f32, height: f32) {
    if points.is_empty() {
        return;
    }

    let mut min_x = points[0].x;
    let mut max_x = points[0].x;
    let mut min_y = points[0].y;
    let mut max_y = points[0].y;

    for p in points.iter().skip(1) {
        if p.x < min_x {
            min_x = p.x;
        }
        if p.x > max_x {
            max_x = p.x;
        }
        if p.y < min_y {
            min_y = p.y;
        }
        if p.y > max_y {
            max_y = p.y;
        }
    }

    let bbox_w = max_x - min_x;
    let bbox_h = max_y - min_y;

    if bbox_w <= f32::EPSILON || bbox_h <= f32::EPSILON {
        return;
    }

    let scale_x = width / bbox_w;
    let scale_y = height / bbox_h;
    let scale = scale_x.min(scale_y);

    let new_width = bbox_w * scale;
    let new_height = bbox_h * scale;

    let offset_x = (width - new_width).mul_add(0.5, -min_x * scale);
    let offset_y = (height - new_height).mul_add(0.5, -min_y * scale);

    for p in points.iter_mut() {
        p.x = p.x.mul_add(scale, offset_x);
        p.y = p.y.mul_add(scale, offset_y);
    }
}

fn koch_fractal(vertices: &[Point], iterations: u8) -> Vec<Point> {
    if iterations == 0 {
        return vertices.to_vec();
    }

    let mut current_vertices = vertices.to_vec();

    for _ in 0..iterations {
        let mut next_vertices = Vec::with_capacity(current_vertices.len() * 4);
        let len = current_vertices.len();

        for i in 0..len {
            let p0 = current_vertices[i];
            let p1 = current_vertices[(i + 1) % len];

            let dx = p1.x - p0.x;
            let dy = p1.y - p0.y;

            let p_a = Point {
                x: p0.x + dx / 3.0,
                y: p0.y + dy / 3.0,
            };
            let p_c = Point {
                x: (dx / 3.0).mul_add(2.0, p0.x),
                y: (dy / 3.0).mul_add(2.0, p0.y),
            };

            // Calculate the peak of the equilateral triangle
            // Vector from p_a to p_c is (dx/3, dy/3).
            // Rotate -60 degrees (outward for CCW polygon)
            let v_x = p_c.x - p_a.x;
            let v_y = p_c.y - p_a.y;

            let sin60 = (PI / 3.0).sin();
            let cos60 = 0.5;

            let p_b_x = p_a.x + v_y.mul_add(sin60, v_x * cos60);
            let p_b_y = p_a.y + v_y.mul_add(cos60, -v_x * sin60);

            let p_b = Point { x: p_b_x, y: p_b_y };

            next_vertices.push(p0);
            next_vertices.push(p_a);
            next_vertices.push(p_b);
            next_vertices.push(p_c);
        }
        current_vertices = next_vertices;
    }

    current_vertices
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
            let angle = delta.mul_add(i as f32, start);
            points.push(Point {
                x: angle.cos().mul_add(radius, cx),
                y: angle.sin().mul_add(radius, cy),
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
            x: angle.cos().mul_add(radius, cx),
            y: angle.sin().mul_add(radius, cy),
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
