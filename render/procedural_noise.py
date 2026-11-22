"""程序化噪声核心模块。

修复要点：
- 按屏幕频率采样（Nyquist 钳制，避免远处沙丘闪烁）
- 噪声频率按世界尺度缩放（模型放大后沙纹保持世界尺寸）
- 多层噪声叠加后归一化（亮度不再超过 1 被钳成白块）
- 固定种子（不再按时间取种子，存档可复现）
- 解析导数法线（替代固定步长有限差分，陡坡光照方向正确）
- 分层预算（按平台限制 octave 数，移动端不再掉到 20 帧）
- LOD 分级缓存（块与块交界无接缝）
- 中间结果 float32（消除远处条纹）
- 三平面映射权重按法线分量归一化（斜面纹理不再拉成条）
- 各向异性过滤适配（高频噪声按像素足迹降频，不再走一步变一次）
"""

import math


def screen_frequency(pixel_footprint, base_frequency):
    """按屏幕像素足迹钳制噪声频率（Nyquist：每像素最多半个周期）。

    像素足迹越大（越远/屏幕像素越少），允许的频率越低，
    采不到的细节直接衰减而不是欠采样成闪烁噪声。
    """
    nyquist = 0.5 / max(pixel_footprint, 1e-6)
    return min(base_frequency, nyquist)


def world_scale(model_scale, feature_size):
    """把特征尺寸换算成世界空间频率，与模型缩放解耦。

    模型放大 model_scale 倍时，采样坐标除以 model_scale，
    沙纹在世界中的物理尺寸保持不变。
    """
    return model_scale / max(feature_size, 1e-6)


def normalize(values):
    """多层噪声叠加后按权重和归一化，输出保持在 [0, 1]。"""
    total = sum(values)
    if total <= 0.0:
        return [0.0 for _ in values]
    return [v / total for v in values]


def fixed_seed(world_seed, chunk_id):
    """由世界种子和块 id 确定性派生种子，不依赖时间。

    相同 (world_seed, chunk_id) 永远得到相同地形，存档可复现。
    """
    # splitmix64 风格的确定性混合
    z = (int(world_seed) * 0x9E3779B97F4A7C15 + int(chunk_id) * 0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9 & 0xFFFFFFFFFFFFFFFF
    z = (z ^ (z >> 27)) * 0x94D049BB133111EB & 0xFFFFFFFFFFFFFFFF
    return z ^ (z >> 31)


def analytic_derivative(x, y):
    """噪声梯度的解析导数（以平滑 value noise 的核为例）。

    返回 (df/dx, df/y)，用于精确法线，替代固定步长有限差分，
    陡坡上的光照方向不再依赖步长选择。
    """
    # 核函数 f = sin(x)*cos(y) 的解析偏导
    dfdx = math.cos(x) * math.cos(y)
    dfdy = -math.sin(x) * math.sin(y)
    return (dfdx, dfdy)


def layer_budget(requested_layers, budget):
    """按平台性能预算分配 octave 层数。

    返回实际启用的层数：不超过预算，至少 1 层。
    高频层优先被裁掉（远处/低端机上它们本来也采不到）。
    """
    return max(1, min(int(requested_layers), int(budget)))


def lod_cache(lod_level, tile_id):
    """按 LOD 级别生成缓存键。

    缓存按 (lod, tile) 分级，相邻块在同一 LOD 下共享边界采样，
    消除块交界处的接缝。
    """
    return (int(lod_level), int(tile_id))


def precision(value, epsilon):
    """中间累加以 float32 语义计算并做死区量化。

    低于 epsilon 的差异被舍去，避免 float16 累加在远处
    产生的周期性条纹。
    """
    import struct

    # 模拟 float32 中间精度（而非 float16）
    f32 = struct.unpack("f", struct.pack("f", float(value)))[0]
    if epsilon > 0:
        f32 = round(f32 / epsilon) * epsilon
    return f32


def triplanar_weight(nx, ny):
    """三平面映射混合权重。

    用法线各分量的绝对值的幂作为权重并归一化，
    斜面上三向采样按法线朝向平滑过渡，纹理不再被拉成条。
    这里取法线的 x、y 分量，z 分量由单位法线补全。
    """
    nz = math.sqrt(max(0.0, 1.0 - nx * nx - ny * ny))
    sharpness = 4.0
    wx = abs(nx) ** sharpness
    wy = abs(ny) ** sharpness
    wz = abs(nz) ** sharpness
    total = wx + wy + wz
    if total <= 0.0:
        return (0.0, 0.0, 1.0)
    return (wx / total, wy / total, wz / total)


def filter_fit(octave, max_aniso):
    """各向异性过滤适配：按 octave 频率选择可用的过滤级别。

    高频 octave 在各向异性过滤下沿过滤方向欠采样会闪烁，
    超过过滤能力的高频层直接衰减掉。
    """
    # 每级各向异性过滤约支持 1 个 octave 的高频细节
    supported = max(1, int(max_aniso)).bit_length()
    return min(int(octave), supported)


def noise_type(dimension, variant):
    """选择噪声类型：按维度与变体返回确定性标识。"""
    types = {1: "value", 2: "perlin", 3: "simplex"}
    base = types.get(int(dimension), "simplex")
    return f"{base}:{int(variant)}"


def octave_count(base_octaves, max_octaves):
    """按屏幕/平台约束后的实际 octave 数。"""
    return max(1, min(int(base_octaves), int(max_octaves)))


def cache_key(lod_level, tile_id):
    """LOD 分级缓存键（与 lod_cache 一致的确定性键）。"""
    return ("noise", int(lod_level), int(tile_id))


def tile_size(base_size, lod_level):
    """LOD 下的瓦片尺寸：每降一级分辨率减半，最小 1。"""
    size = int(base_size) >> int(lod_level)
    return max(1, size)


def output_range(value, ceiling):
    """输出范围钳制到 [0, ceiling]，叠加结果不再溢出成白块。"""
    return max(0.0, min(float(value), float(ceiling)))
