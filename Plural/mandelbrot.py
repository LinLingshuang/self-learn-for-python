#!/usr/bin/env python3
"""生成并显示/保存经典曼德勃罗集合的脚本。

用法示例：
    python mandelbrot.py --width 1200 --height 800 --max-iter 500 --output mandelbrot.png
    e:/Code/Github/self-learn-for-python/Plural/.venv/Scripts/python.exe mandelbrot.py --interactive
默认会弹窗显示图像；指定 `--output` 可保存为文件。
"""
from __future__ import annotations

import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional


def mandelbrot(width: int = 800,
               height: int = 600,
               x_min: float = -2.0,
               x_max: float = 1.0,
               y_min: float = -1.2,
               y_max: float = 1.2,
               max_iter: int = 200,
               initial_z: complex = 0+0j) -> np.ndarray:
    """计算曼德勃罗集合的迭代逃逸时间阵列。

    返回值为 shape (height, width) 的整数数组，表示每个像素在第几次迭代时逃逸（达到 |z| > 2 ）。
    未逃逸的点值为 max_iter。
    """
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)
    C = X + 1j * Y

    Z = np.full(C.shape, initial_z, dtype=np.complex128)
    div_time = np.zeros(C.shape, dtype=int)
    mask = np.ones(C.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] * Z[mask] + C[mask]
        mask_now = np.abs(Z) <= 2.0
        escaped = mask & (~mask_now)
        div_time[escaped] = i
        mask = mask_now

    div_time[mask] = max_iter
    return div_time


def julia(c: complex,
          width: int = 800,
          height: int = 600,
          x_min: float = -1.6,
          x_max: float = 1.6,
          y_min: float = -1.2,
          y_max: float = 1.2,
          max_iter: int = 200,
          initial_z: complex = 0+0j) -> np.ndarray:
    """计算固定参数 c 的 Julia 集合逃逸时间阵列。"""
    x = np.linspace(x_min, x_max, width)
    y = np.linspace(y_min, y_max, height)
    X, Y = np.meshgrid(x, y)

    # Julia 中每个像素对应 z0；保留额外偏移 initial_z 便于实验不同起始值。
    Z = (X + 1j * Y + initial_z).astype(np.complex128)
    div_time = np.zeros(Z.shape, dtype=int)
    mask = np.ones(Z.shape, dtype=bool)

    for i in range(max_iter):
        Z[mask] = Z[mask] * Z[mask] + c
        mask_now = np.abs(Z) <= 2.0
        escaped = mask & (~mask_now)
        div_time[escaped] = i
        mask = mask_now

    div_time[mask] = max_iter
    return div_time


def plot_mandelbrot(data: np.ndarray,
                    x_min: float,
                    x_max: float,
                    y_min: float,
                    y_max: float,
                    cmap: str = 'twilight_shifted',
                    output: Optional[str] = None,
                    dpi: int = 100) -> None:
    plt.figure(figsize=(data.shape[1] / dpi, data.shape[0] / dpi), dpi=dpi)
    extent = (x_min, x_max, y_min, y_max)
    plt.imshow(data, extent=extent, cmap=cmap, origin='lower')
    plt.xlabel('Re')
    plt.ylabel('Im')
    plt.title('Mandelbrot Set')
    plt.colorbar(label='Iteration (escape time)')
    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=dpi)
        print(f"Saved image to {output}")
    else:
        plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(description='Render the Mandelbrot set')
    parser.add_argument('--width', type=int, default=800, help='image width in pixels')
    parser.add_argument('--height', type=int, default=600, help='image height in pixels')
    parser.add_argument('--x-min', type=float, default=-2.0)
    parser.add_argument('--x-max', type=float, default=1.0)
    parser.add_argument('--y-min', type=float, default=-1.2)
    parser.add_argument('--y-max', type=float, default=1.2)
    parser.add_argument('--max-iter', type=int, default=200, help='maximum iterations')
    parser.add_argument('--cmap', type=str, default='twilight_shifted', help='matplotlib colormap')
    parser.add_argument('--output', type=str, default=None, help='output filename (PNG, PDF, ...). If omitted, display on screen')
    parser.add_argument('--dpi', type=int, default=100, help='figure DPI when saving')
    parser.add_argument('--initial-z', type=str, default=None, help='initial z value, e.g. "0.3+0.5j" (default 0)')
    parser.add_argument('--interactive', action='store_true', help='enable interactive mode: click Mandelbrot to switch to corresponding Julia set')

    args = parser.parse_args()

    initial_z = 0 + 0j
    if args.initial_z is not None:
        try:
            initial_z = complex(args.initial_z)
        except Exception:
            print('Could not parse --initial-z, using 0+0j')

    def draw_image(data: np.ndarray, title: str) -> None:
        plt.clf()
        extent = (args.x_min, args.x_max, args.y_min, args.y_max)
        plt.imshow(data, extent=extent, cmap=args.cmap, origin='lower')
        plt.xlabel('Re')
        plt.ylabel('Im')
        plt.title(title)
        plt.colorbar(label='Iteration (escape time)')
        plt.tight_layout()
        plt.draw()

    def render_mandelbrot(init_z: complex) -> np.ndarray:
        return mandelbrot(width=args.width,
                          height=args.height,
                          x_min=args.x_min,
                          x_max=args.x_max,
                          y_min=args.y_min,
                          y_max=args.y_max,
                          max_iter=args.max_iter,
                          initial_z=init_z)

    def render_julia(c_value: complex, init_z: complex) -> np.ndarray:
        return julia(c=c_value,
                     width=args.width,
                     height=args.height,
                     x_min=args.x_min,
                     x_max=args.x_max,
                     y_min=args.y_min,
                     y_max=args.y_max,
                     max_iter=args.max_iter,
                     initial_z=init_z)

    if args.interactive:
        fig = plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi)
        view_mode = 'mandelbrot'

        mandelbrot_data = render_mandelbrot(initial_z)
        draw_image(mandelbrot_data, f'Mandelbrot Set (initial z={initial_z})')

        def on_click(event):
            if event.xdata is None or event.ydata is None:
                return

            nonlocal view_mode

            # 鼠标左键：在 Mandelbrot 图中点击某点 c，切换到对应 Julia(c)。
            if event.button == 1 and view_mode == 'mandelbrot':
                c_value = complex(event.xdata, event.ydata)
                julia_data = render_julia(c_value, initial_z)
                draw_image(julia_data, f'Julia Set (c={c_value}, initial z offset={initial_z})')
                view_mode = 'julia'
                print(f'Switched to Julia set with c={c_value}. Right click to return to Mandelbrot.')
                return

            # 鼠标右键：从 Julia 图返回 Mandelbrot 图。
            if event.button == 3 and view_mode == 'julia':
                mandelbrot_data_local = render_mandelbrot(initial_z)
                draw_image(mandelbrot_data_local, f'Mandelbrot Set (initial z={initial_z})')
                view_mode = 'mandelbrot'
                print('Returned to Mandelbrot set.')
                return

            # 在 Julia 模式左键可直接选取新 c 并刷新 Julia。
            if event.button == 1 and view_mode == 'julia':
                c_value = complex(event.xdata, event.ydata)
                julia_data = render_julia(c_value, initial_z)
                draw_image(julia_data, f'Julia Set (c={c_value}, initial z offset={initial_z})')
                print(f'Updated Julia set to c={c_value}. Right click to return to Mandelbrot.')

        fig.canvas.mpl_connect('button_press_event', on_click)
        print('Interactive mode: left click Mandelbrot -> Julia(c), right click Julia -> Mandelbrot.')
        plt.show()
    else:
        data = render_mandelbrot(initial_z)
        plt.figure(figsize=(args.width / args.dpi, args.height / args.dpi), dpi=args.dpi)
        draw_image(data, f'Mandelbrot Set (initial z={initial_z})')

        if args.output:
            plt.savefig(args.output, dpi=args.dpi)
            print(f"Saved image to {args.output}")
        else:
            plt.show()


if __name__ == '__main__':
    main()
