//! pic2color4px — convert a photo to an 8-colour image (RGB on/off) using
//! the 2x2-cell pattern-dither method of pic2gray4px, applied per channel
//! and optimised jointly.
//!
//! Each 2x2 cell carries, per channel, a density of 0..4 dots, so the eye
//! integrates 5 levels per channel (~125 apparent colours) while the stored
//! image uses exactly 8 colours. That is the palette of MSX-era hardware,
//! and the same trick colour e-paper uses with its pigments.
//!
//! Usage:
//!   pic2color4px <input.png|jpg> [--width N] [-o output.png] [--force]
//! Default output: <input-stem>-color4px.png next to the input.

use std::env;
use std::path::{Path, PathBuf};
use std::process::ExitCode;

// patterns: bit i = pixel (x = i % 2, y = i / 2) of the 2x2 cell
fn patterns(k: u8) -> &'static [u8] {
    match k {
        0 => &[0b0000],
        1 => &[0b0001, 0b0010, 0b0100, 0b1000],
        2 => &[0b1001, 0b0110],
        3 => &[0b1110, 0b1101, 0b1011, 0b0111],
        _ => &[0b1111],
    }
}
const POS: [(usize, usize); 4] = [(0, 0), (1, 0), (0, 1), (1, 1)];

fn gauss1d() -> [f32; 9] {
    let sigma = 1.1f32;
    let mut g = [0f32; 9];
    let mut sum = 0f32;
    for i in 0..9 {
        let x = i as f32 - 4.0;
        g[i] = (-x * x / (2.0 * sigma * sigma)).exp();
        sum += g[i];
    }
    for v in g.iter_mut() {
        *v /= sum;
    }
    g
}

fn reflect(i: isize, n: usize) -> usize {
    let n = n as isize;
    let mut i = i;
    while i < 0 || i >= n {
        if i < 0 {
            i = -i;
        }
        if i >= n {
            i = 2 * n - 2 - i;
        }
    }
    i as usize
}

/// Separable Gaussian blur of an HxWx3 f32 image.
fn blur(img: &[f32], w: usize, h: usize, g: &[f32; 9]) -> Vec<f32> {
    let r = 4isize;
    let mut tmp = vec![0f32; w * h * 3];
    let mut out = vec![0f32; w * h * 3];
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut acc = 0f32;
                for k in -r..=r {
                    let sx = reflect(x as isize + k, w);
                    acc += g[(k + r) as usize] * img[(y * w + sx) * 3 + c];
                }
                tmp[(y * w + x) * 3 + c] = acc;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            for c in 0..3 {
                let mut acc = 0f32;
                for k in -r..=r {
                    let sy = reflect(y as isize + k, h);
                    acc += g[(k + r) as usize] * tmp[(sy * w + x) * 3 + c];
                }
                out[(y * w + x) * 3 + c] = acc;
            }
        }
    }
    out
}

/// Cell counts from cell means via cell-level error diffusion (per channel,
/// Floyd–Steinberg weights transplanted to the cell grid).
fn cell_counts(means: &[f32], wc: usize, hc: usize) -> Vec<[u8; 3]> {
    let mut err = vec![[0f32; 3]; wc * hc];
    let mut counts = vec![[0u8; 3]; wc * hc];
    const WTS: [((isize, isize), f32); 4] = [
        ((1, 0), 7.0 / 16.0),
        ((-1, 1), 3.0 / 16.0),
        ((0, 1), 5.0 / 16.0),
        ((1, 1), 1.0 / 16.0),
    ];
    for y in 0..hc {
        for x in 0..wc {
            let i = y * wc + x;
            let mut k = [0u8; 3];
            let mut e = [0f32; 3];
            for c in 0..3 {
                let t = means[i * 3 + c] + err[i][c];
                let kk = (t * 4.0 + 0.5).floor().clamp(0.0, 4.0) as u8;
                k[c] = kk;
                e[c] = (t * 4.0 - kk as f32) / 4.0;
            }
            counts[i] = k;
            for ((dx, dy), wgt) in WTS {
                let nx = x as isize + dx;
                let ny = y as isize + dy;
                if nx >= 0 && nx < wc as isize && ny >= 0 && ny < hc as isize {
                    let ni = ny as usize * wc + nx as usize;
                    for c in 0..3 {
                        err[ni][c] += e[c] * wgt;
                    }
                }
            }
        }
    }
    counts
}

/// Joint arrangement optimisation: for every cell and channel, choose the
/// pattern (among those with the decided count) that minimises the
/// perceived colour error — the residual between the Gaussian-blurred
/// output and the Gaussian-blurred original, all three channels together.
/// Deterministic: fixed initial patterns, fixed scan order, fixed passes.
fn optimise(target: &[f32], counts: &[[u8; 3]], w: usize, h: usize) -> Vec<f32> {
    let wc = w / 2;
    let hc = h / 2;
    let g = gauss1d();
    let r = 4isize;
    let mut pats: Vec<[u8; 3]> = counts
        .iter()
        .map(|k| [patterns(k[0])[0], patterns(k[1])[0], patterns(k[2])[0]])
        .collect();
    let mut comp = vec![0f32; w * h * 3];
    for cy in 0..hc {
        for cx in 0..wc {
            let ci = cy * wc + cx;
            for c in 0..3 {
                let m = pats[ci][c];
                for (i, (px, py)) in POS.iter().enumerate() {
                    if (m >> i) & 1 == 1 {
                        comp[((cy * 2 + py) * w + (cx * 2 + px)) * 3 + c] = 1.0;
                    }
                }
            }
        }
    }
    let bt = blur(target, w, h, &g);
    let bc = blur(&comp, w, h, &g);
    let mut res: Vec<f32> = bc.iter().zip(bt.iter()).map(|(a, b)| a - b).collect();

    for _pass in 0..8 {
        let mut changed = 0usize;
        for cy in 0..hc {
            for cx in 0..wc {
                let ci = cy * wc + cx;
                for c in 0..3 {
                    let cur = pats[ci][c];
                    let k = counts[ci][c];
                    let mut best = cur;
                    let mut best_gain = 0f32;
                    for &m in patterns(k) {
                        if m == cur {
                            continue;
                        }
                        let mut deltas: [(usize, usize, f32); 4] = [(0, 0, 0.0); 4];
                        let mut nd = 0usize;
                        let (mut x0, mut y0) = (usize::MAX, usize::MAX);
                        let (mut x1, mut y1) = (0usize, 0usize);
                        for (i, (px, py)) in POS.iter().enumerate() {
                            let d = ((m >> i) & 1) as f32 - ((cur >> i) & 1) as f32;
                            if d != 0.0 {
                                let gx = cx * 2 + px;
                                let gy = cy * 2 + py;
                                deltas[nd] = (gx, gy, d);
                                nd += 1;
                                x0 = x0.min(gx.saturating_sub(4));
                                y0 = y0.min(gy.saturating_sub(4));
                                x1 = x1.max((gx + 4).min(w - 1));
                                y1 = y1.max((gy + 4).min(h - 1));
                            }
                        }
                        if nd == 0 {
                            continue;
                        }
                        let fw = x1 - x0 + 1;
                        let fh = y1 - y0 + 1;
                        let mut db = vec![0f32; fw * fh];
                        for &(gx, gy, d) in deltas.iter().take(nd) {
                            for wy in y0..=y1 {
                                let ky = wy as isize - gy as isize + r;
                                if !(0..9).contains(&ky) {
                                    continue;
                                }
                                for wx in x0..=x1 {
                                    let kx = wx as isize - gx as isize + r;
                                    if (0..9).contains(&kx) {
                                        db[(wy - y0) * fw + (wx - x0)] +=
                                            d * g[ky as usize] * g[kx as usize];
                                    }
                                }
                            }
                        }
                        let mut s = 0f32;
                        for wy in y0..=y1 {
                            for wx in x0..=x1 {
                                let dbv = db[(wy - y0) * fw + (wx - x0)];
                                if dbv != 0.0 {
                                    let rv = res[(wy * w + wx) * 3 + c];
                                    s += 2.0 * rv * dbv + dbv * dbv;
                                }
                            }
                        }
                        let gain = -s;
                        if gain > best_gain + 1e-9 {
                            best_gain = gain;
                            best = m;
                        }
                    }
                    if best != cur {
                        for (i, (px, py)) in POS.iter().enumerate() {
                            let d = ((best >> i) & 1) as f32 - ((cur >> i) & 1) as f32;
                            if d != 0.0 {
                                let gx = cx * 2 + px;
                                let gy = cy * 2 + py;
                                comp[(gy * w + gx) * 3 + c] += d;
                                let xa = gx.saturating_sub(4);
                                let ya = gy.saturating_sub(4);
                                let xb = (gx + 4).min(w - 1);
                                let yb = (gy + 4).min(h - 1);
                                for wy in ya..=yb {
                                    for wx in xa..=xb {
                                        res[(wy * w + wx) * 3 + c] += d
                                            * g[(wy as isize - gy as isize + r) as usize]
                                            * g[(wx as isize - gx as isize + r) as usize];
                                    }
                                }
                            }
                        }
                        pats[ci][c] = best;
                        changed += 1;
                    }
                }
            }
        }
        if changed == 0 {
            break;
        }
    }
    comp
}

fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    let mut input: Option<String> = None;
    let mut output: Option<String> = None;
    let mut width: Option<u32> = None;
    let mut force = false;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--width" => {
                i += 1;
                width = Some(args.get(i).ok_or("--width needs a value")?.parse()?);
            }
            "-o" | "--output" => {
                i += 1;
                output = Some(args.get(i).ok_or("-o needs a value")?.clone());
            }
            "--force" => force = true,
            "-h" | "--help" => {
                println!(
                    "usage: pic2color4px <input> [--width N] [-o output.png] [--force]"
                );
                return Ok(());
            }
            other => {
                if input.is_some() {
                    return Err(format!("unexpected argument: {other}").into());
                }
                input = Some(other.to_string());
            }
        }
        i += 1;
    }
    let input = input.ok_or("usage: pic2color4px <input> [--width N] [-o output.png] [--force]")?;

    let mut img = image::open(&input)?.to_rgb8();
    if let Some(w) = width {
        let (ow, oh) = (img.width(), img.height());
        let nh = ((oh as f32 * w as f32 / ow as f32).round() as u32).max(2);
        img = image::imageops::resize(&img, w, nh, image::imageops::FilterType::Lanczos3);
    }
    let (w, h) = (img.width() as usize, img.height() as usize);
    let pw = w.div_ceil(2) * 2;
    let ph = h.div_ceil(2) * 2;
    let (wc, hc) = (pw / 2, ph / 2);

    // full-res target (edge-padded) and cell means
    let mut target = vec![0f32; pw * ph * 3];
    let mut means = vec![0f32; wc * hc * 3];
    for y in 0..ph {
        for x in 0..pw {
            let sx = x.min(w - 1);
            let sy = y.min(h - 1);
            let p = img.get_pixel(sx as u32, sy as u32);
            for c in 0..3 {
                target[(y * pw + x) * 3 + c] = p[c] as f32 / 255.0;
            }
        }
    }
    for cy in 0..hc {
        for cx in 0..wc {
            for c in 0..3 {
                let mut acc = 0f32;
                for (px, py) in POS {
                    acc += target[((cy * 2 + py) * pw + (cx * 2 + px)) * 3 + c];
                }
                means[(cy * wc + cx) * 3 + c] = acc / 4.0;
            }
        }
    }

    let counts = cell_counts(&means, wc, hc);
    let comp = optimise(&target, &counts, pw, ph);

    let out_path: PathBuf = match output {
        Some(o) => PathBuf::from(o),
        None => {
            let p = Path::new(&input);
            let stem = p.file_stem().ok_or("bad input path")?.to_string_lossy();
            p.with_file_name(format!("{stem}-color4px.png"))
        }
    };
    if out_path.exists() && !force {
        return Err(format!(
            "{} already exists (use --force to overwrite)",
            out_path.display()
        )
        .into());
    }
    let mut out = image::RgbImage::new(w as u32, h as u32);
    for y in 0..h {
        for x in 0..w {
            let j = (y * pw + x) * 3;
            out.put_pixel(
                x as u32,
                y as u32,
                image::Rgb([
                    (comp[j] * 255.0) as u8,
                    (comp[j + 1] * 255.0) as u8,
                    (comp[j + 2] * 255.0) as u8,
                ]),
            );
        }
    }
    out.save(&out_path)?;
    println!(
        "converted {input} ({w}x{h}) -> {} (8 colours)",
        out_path.display()
    );
    Ok(())
}

fn main() -> ExitCode {
    match run() {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            eprintln!("error: {e}");
            ExitCode::FAILURE
        }
    }
}
