import numpy as np

from pptt.visualization.method_overview import render_pptt_method_figure


def test_method_overview_renders_continuous_graphical_pipeline(tmp_path):
    image = np.zeros((48, 48), dtype=np.float32)
    image[8:40, 8:40] = np.linspace(0.0, 1.0, 32 * 32).reshape(32, 32)
    truth = np.zeros((48, 48), dtype=np.uint8)
    truth[15:35, 15:35] = 2
    states = np.zeros((8, 48, 48), dtype=np.uint8)
    states[2:5, 18:32, 18:32] = 1
    states[5:, 15:35, 15:35] = 2
    reliable = np.ones((7, 48, 48), dtype=bool)

    png, pdf = render_pptt_method_figure(
        output_base=tmp_path / "method",
        language="en",
        image=image,
        truth=truth,
        states=states,
        reliable=reliable,
        final_state=states[-1],
        dpi=300,
    )

    assert png.is_file() and png.stat().st_size > 8000
    assert pdf.is_file() and pdf.stat().st_size > 1000
