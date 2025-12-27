//! Shape utilities for custom crop geometry and masking.
//!
//! Provides the `CropShape` enum shared across the workspace together with helpers
//! for generating polygon outlines and applying alpha masks to RGBA images.

use crate::point::Point;

use image::{DynamicImage, RgbaImage};
use serde::{Deserialize, Serialize};
use std::f32::consts::PI;
use tiny_skia::{FillRule, Paint, PathBuilder, Pixmap, Transform};

/// Polygon corner styles.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(tag = "style", rename_all = "snake_case")]
pub enum PolygonCornerStyle {
    #[default]
    Sharp,
    Rounded {
        radius_pct: f32,
    },
    Chamfered {
        size_pct: f32,
    },
    Bezier {
        tension: f32,
    },
}

/// Shapes supported by the crop exporter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Default)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum CropShape {
    #[default]
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
            let cx = w * 0.5;
            let cy = h * 0.5;
            let segments = 128;
            (0..segments)
                .map(|i| {
                    let theta = (i as f32 / segments as f32) * 2.0 * PI;
                    Point {
                        x: theta.cos().mul_add(cx, cx),
                        y: theta.sin().mul_add(cy, cy),
                    }
                })
                .collect()
        }
        CropShape::RoundedRectangle { radius_pct } => {
            let radius = (w.min(h) * radius_pct).clamp(0.0, w.min(h) * 0.5);
            rounded_rect_points(w, h, radius, 16)
        }
        CropShape::ChamferedRectangle { size_pct } => {
            let inset = (w.min(h) * size_pct).clamp(0.0, w.min(h) * 0.5);
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
    let mut min_points = points[0];
    let mut max_points = points[0];

    for p in points.iter().skip(1) {
        if p.x < min_points.x {
            min_points.x = p.x;
        }
        if p.x > max_points.x {
            max_points.x = p.x;
        }
        if p.y < min_points.y {
            min_points.y = p.y;
        }
        if p.y > max_points.y {
            max_points.y = p.y;
        }
    }

    let bbox = max_points - min_points;

    if bbox.x <= f32::EPSILON || bbox.y <= f32::EPSILON {
        return;
    }

    let scale_x = width / bbox.x;
    let scale_y = height / bbox.y;
    let scale = scale_x.min(scale_y);

    let new_width = bbox.x * scale;
    let new_height = bbox.y * scale;

    let offset_x = (width - new_width).mul_add(0.5, -min_points.x * scale);
    let offset_y = (height - new_height).mul_add(0.5, -min_points.y * scale);

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
            let dxy = p1 - p0;
            let p_a = p0 + dxy / 3.0;
            let p_c = (dxy / 3.0).mul_add(2.0, p0);

            // Calculate the peak of the equilateral triangle
            // Vector from p_a to p_c is (dx/3, dy/3).
            // Rotate -60 degrees (outward for CCW polygon)
            let v = p_c - p_a;

            let sin60 = (PI / 3.0).sin();
            let cos60 = 0.5;

            let p_b_x = p_a.x + v.y.mul_add(sin60, v.x * cos60);
            let p_b_y = p_a.y + v.y.mul_add(cos60, -v.x * sin60);

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
            push_point(&mut points, angle, cx, cy, radius);
        }
    };

    // Top-right corner (angles -90 -> 0 degrees)
    add_corner(width - radius, radius, -PI * 0.5, 0.0);
    // Bottom-right (0 -> 90)
    add_corner(width - radius, height - radius, 0.0, PI * 0.5);
    // Bottom-left (90 -> 180)
    add_corner(radius, height - radius, PI * 0.5, PI);
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
    let cx = width * 0.5;
    let cy = height * 0.5;
    let radius = 0.5 * width.min(height);
    let rotation = rotation_deg.to_radians();

    let mut base_vertices = Vec::with_capacity(n);
    for i in 0..n {
        let angle = rotation + 2.0 * PI * i as f32 / n as f32;
        push_point(&mut base_vertices, angle, cx, cy, radius);
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

        let prev_vec = normalize(current - prev);
        let next_vec = normalize(next - current);

        let prev_edge_len = distance(prev, current);
        let next_edge_len = distance(current, next);
        let offset_prev = inset.min(prev_edge_len * 0.5);
        let offset_next = inset.min(next_edge_len * 0.5);

        points.push((-prev_vec).mul_add(offset_prev, current));
        points.push(next_vec.mul_add(offset_next, current));
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

        let incoming = normalize(current - prev);
        let outgoing = normalize(next - current);

        let angle_cos = (-incoming) * outgoing;
        let angle_cos = angle_cos.clamp(-0.999_9, 0.999_9);
        let half_angle = 0.5 * angle_cos.acos();
        let mut offset = radius / half_angle.tan();
        let incoming_len = distance(prev, current);
        let outgoing_len = distance(current, next);
        offset = offset.min(incoming_len * 0.5).min(outgoing_len * 0.5);

        let start = (-incoming).mul_add(offset, current);
        let end = outgoing.mul_add(offset, current);

        let bisector = normalize(outgoing - incoming);
        let center_distance = radius / half_angle.sin();

        let center = bisector.mul_add(center_distance, current);

        let start_angle = (start.y - center.y).atan2(start.x - center.x);
        let end_angle = (end.y - center.y).atan2(end.x - center.x);
        let mut delta = end_angle - start_angle;
        while delta <= 0.0 {
            delta += 2.0 * PI;
        }
        let steps = segments.max(3);
        let step = delta / steps as f32;
        for j in 0..=steps {
            let angle = step.mul_add(j as f32, start_angle);
            push_point(&mut points, angle, center.x, center.y, radius);
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
        let tangent = next - prev;

        // Scale by tension
        // tension of 0.5 is standard Catmull-Rom
        let cp_dist = tension * 0.5; // Adjust scaling factor as needed

        // Control point "before" current (incoming)
        let cp1 = (-tangent).mul_add(cp_dist, current);
        // Control point "after" current (outgoing)
        let cp2 = tangent.mul_add(cp_dist, current);

        control_points.push((cp1, cp2));
    }

    // Generate curve segments between vertices
    for i in 0..len {
        let p0 = vertices[i];
        let p1 = vertices[(i + 1) % len];

        // Control points for this segment:
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

    p0 * mt3 + p1 * 3.0 * mt2 * t + p2 * 3.0 * mt * t2 + p3 * t3
}

#[inline]
fn distance(a: Point, b: Point) -> f32 {
    ((a.x - b.x).powi(2) + (a.y - b.y).powi(2)).sqrt()
}

fn normalize(v: Point) -> Point {
    let len = (v.x * v.x + v.y * v.y).sqrt();
    if len <= f32::EPSILON {
        Point { x: 0.0, y: 0.0 }
    } else {
        v / len
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
pub fn apply_shape_mask(
    image: &mut RgbaImage,
    shape: &CropShape,
    vignette_softness: f32,
    vignette_intensity: f32,
    vignette_color: crate::color::RgbaColor,
) {
    if matches!(shape, CropShape::Rectangle) && vignette_softness <= 0.0 {
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

        if vignette_softness > 0.0 {
            let radius = (width.min(height) as f32 * 0.5 * vignette_softness).max(1.0);
            let blurred = image::imageops::blur(
                &RgbaImage::from_raw(width, height, pixmap.data().to_vec()).unwrap(),
                radius,
            );
            let mask = blurred.as_raw();
            for (pixel, mask_px) in image.pixels_mut().zip(mask.chunks_exact(4)) {
                let mask_alpha = mask_px[3] as f32 / 255.0;

                // Vignette color mixing: blend toward vignette color in masked-out regions
                let inv_mask = 1.0 - mask_alpha;
                for c in 0..3 {
                    let img_val = pixel[c] as f32;
                    let vig_val = match c {
                        0 => vignette_color.red as f32,
                        1 => vignette_color.green as f32,
                        2 => vignette_color.blue as f32,
                        _ => 0.0,
                    };
                    // Blend image with vignette color based on mask and intensity
                    pixel[c] = (img_val + inv_mask * vignette_intensity * (vig_val - img_val))
                        .clamp(0.0, 255.0) as u8;
                }

                // Alpha: purely based on mask, independent of vignette intensity
                pixel[3] = (pixel[3] as f32 * mask_alpha).clamp(0.0, 255.0) as u8;
            }
        } else {
            let mask = pixmap.data();
            for (pixel, mask_px) in image.pixels_mut().zip(mask.chunks_exact(4)) {
                let mask_alpha = mask_px[3] as f32 / 255.0;

                // Outside the shape: apply vignette color and make transparent
                if mask_alpha < 0.5 {
                    // Mix with vignette color based on intensity
                    for c in 0..3 {
                        let img_val = pixel[c] as f32;
                        let vig_val = match c {
                            0 => vignette_color.red as f32,
                            1 => vignette_color.green as f32,
                            2 => vignette_color.blue as f32,
                            _ => 0.0,
                        };
                        pixel[c] = (img_val + vignette_intensity * (vig_val - img_val))
                            .clamp(0.0, 255.0) as u8;
                    }
                    // Alpha: outside is always transparent (clean crop)
                    pixel[3] = 0;
                }
            }
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
    let cx = width * 0.5;
    let cy = height * 0.5;
    let outer_radius = 0.5 * width.min(height);
    let inner_radius = outer_radius * inner_radius_pct;
    let rotation = rotation_deg.to_radians();

    let mut vertices = Vec::with_capacity(n * 2);
    let step_angle = PI / n as f32;

    for i in 0..n {
        // Outer point
        let angle_outer = rotation + 2.0 * PI * i as f32 / n as f32;
        push_point(&mut vertices, angle_outer, cx, cy, outer_radius);

        // Inner point
        let angle_inner = angle_outer + step_angle;
        push_point(&mut vertices, angle_inner, cx, cy, inner_radius);
    }

    vertices
}

/// Apply the shape mask to a dynamic image, upgrading to RGBA as needed.
pub fn apply_shape_mask_dynamic(
    image: &mut DynamicImage,
    shape: &CropShape,
    vignette_softness: f32,
    vignette_intensity: f32,
    vignette_color: crate::color::RgbaColor,
) {
    if matches!(shape, CropShape::Rectangle) && vignette_softness <= 0.0 {
        return;
    }

    let mut rgba = image.to_rgba8();
    apply_shape_mask(
        &mut rgba,
        shape,
        vignette_softness,
        vignette_intensity,
        vignette_color,
    );
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

/// Pushes a point to the vector, scaled to the given rectangle.
#[inline]
fn push_point(points: &mut Vec<Point>, angle: f32, cx: f32, cy: f32, radius: f32) {
    points.push(Point {
        x: angle.cos().mul_add(radius, cx),
        y: angle.sin().mul_add(radius, cy),
    });
}
