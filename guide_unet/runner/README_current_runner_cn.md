# 当前 runner 说明

5.12 组会后，旧的导出脚本已经归档到：

```text
guide_unet/runner/archive_pre_phenomenon_cause_20260512/
```

当前新 runner 必须服务于“现象原因解释”：

1. 导出 Stage 1 加速/减速曲线与速度、加速度。
2. 导出逐层分割 CAM 轨迹。
3. 计算 CAM 与最终 CAM/GT 的相关性和对齐。
4. 计算 CAM 接近速度、加速度、熵和响应分布。

禁止继续新增以 `skip` 作用排序、结构依赖扩展或模型改进为主目标的 runner。

