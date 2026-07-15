from pathlib import Path

from pptt.experiments.model_matrix import load_model_matrix


def test_model_matrix_resolves_asset_and_workspace_checkpoints(tmp_path: Path):
    asset = tmp_path / "assets"
    workspace = tmp_path / "workspace"
    asset.mkdir()
    workspace.mkdir()
    matrix_path = tmp_path / "matrix.yaml"
    matrix_path.write_text(
        f"""
asset_root:
  environment_variable: TEST_UNUSED_ASSET_ROOT
  default: {asset.as_posix()}
jobs:
  - model: unet_baseline
    seed: 42
    model_config: configs/model.yaml
    checkpoint:
      root: asset
      path: weights/a.pth
  - model: unet_baseline
    seed: 123
    model_config: configs/model.yaml
    checkpoint:
      root: workspace
      path: results/b.pth
""",
        encoding="utf-8",
    )

    matrix = load_model_matrix(matrix_path, workspace_root=workspace)

    assert matrix.asset_root == asset.resolve()
    assert matrix.jobs[0].checkpoint == (asset / "weights/a.pth").resolve()
    assert matrix.jobs[1].checkpoint == (workspace / "results/b.pth").resolve()
    assert matrix.jobs[0].job_id == "unet_baseline/seed_42"
