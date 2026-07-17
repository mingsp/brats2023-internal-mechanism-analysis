import numpy as np

from pptt.transitions.metrics import metrics_from_confusion
from pptt.transitions.tensors import confusions_from_transition


def test_endpoint_metrics_do_not_identify_pixel_transition_process():
    """Equal endpoint metrics can conceal different correction/error events."""
    process_a = np.zeros((3, 3, 3), dtype=np.int64)
    process_b = np.zeros_like(process_a)

    for truth_class in (0, 1):
        process_a[truth_class, truth_class, truth_class] = 1
        process_b[truth_class, truth_class, truth_class] = 1

    process_a[2, 0, 0] = 1
    process_a[2, 1, 1] = 1
    process_a[2, 2, 2] = 1

    process_b[2, 0, 1] = 1
    process_b[2, 1, 2] = 1
    process_b[2, 2, 0] = 1

    before_a, after_a = confusions_from_transition(process_a)
    before_b, after_b = confusions_from_transition(process_b)
    np.testing.assert_array_equal(before_a, before_b)
    np.testing.assert_array_equal(after_a, after_b)

    for confusion_a, confusion_b in ((before_a, before_b), (after_a, after_b)):
        metrics_a = metrics_from_confusion(confusion_a)
        metrics_b = metrics_from_confusion(confusion_b)
        for name in metrics_a.as_dict():
            np.testing.assert_array_equal(
                getattr(metrics_a, name),
                getattr(metrics_b, name),
            )

    assert not np.array_equal(process_a, process_b)
    assert process_a[2, 1, 2] == 0  # no correction in process A
    assert process_b[2, 1, 2] == 1  # correction in process B
    assert process_a[2, 2, 0] == 0  # no destruction in process A
    assert process_b[2, 2, 0] == 1  # destruction in process B
    assert process_a[2, 0, 1] == 0  # no wrong re-encoding in process A
    assert process_b[2, 0, 1] == 1  # wrong re-encoding in process B
