use std::ops::{Add, Div, Mul, Neg, Sub};

/// Single 2D point.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Point {
    pub x: f32,
    pub y: f32,
}

impl Point {
    pub fn mul_add(self, a: f32, b: Point) -> Point {
        Point {
            x: self.x.mul_add(a, b.x),
            y: self.y.mul_add(a, b.y),
        }
    }

    pub fn hypot(self) -> f32 {
        self.x.hypot(self.y)
    }
}

impl Add for Point {
    type Output = Point;

    fn add(self, other: Point) -> Point {
        Point {
            x: self.x + other.x,
            y: self.y + other.y,
        }
    }
}

impl Sub for Point {
    type Output = Point;

    fn sub(self, other: Point) -> Point {
        Point {
            x: self.x - other.x,
            y: self.y - other.y,
        }
    }
}

impl Mul<f32> for Point {
    type Output = Point;

    fn mul(self, other: f32) -> Point {
        Point {
            x: self.x * other,
            y: self.y * other,
        }
    }
}

impl Mul<Point> for Point {
    type Output = f32;

    fn mul(self, other: Point) -> f32 {
        self.x * other.x + self.y * other.y
    }
}

impl Div<f32> for Point {
    type Output = Point;

    fn div(self, other: f32) -> Point {
        Point {
            x: self.x / other,
            y: self.y / other,
        }
    }
}

impl Neg for Point {
    type Output = Point;

    fn neg(self) -> Point {
        Point {
            x: -self.x,
            y: -self.y,
        }
    }
}
