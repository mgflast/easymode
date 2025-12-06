"""Basic import tests to verify package installation."""


def test_import_main():
    """Test that main module can be imported."""
    import easymode.main

    assert hasattr(easymode.main, "main")


def test_import_core():
    """Test that core modules can be imported."""
    import easymode.core.config

    assert hasattr(easymode.core.config, "settings")


def test_import_segmentation():
    """Test that segmentation modules can be imported."""
    import easymode.segmentation.inference

    # Verify module loads
    assert easymode.segmentation.inference is not None


def test_import_ddw():
    """Test that ddw modules can be imported."""
    import easymode.ddw.inference

    assert easymode.ddw.inference is not None


def test_import_n2n():
    """Test that n2n modules can be imported."""
    import easymode.n2n.inference

    assert easymode.n2n.inference is not None


def test_cli_entry_point():
    """Test that CLI entry point exists."""
    from easymode.main import main

    # Entry point exists
    assert callable(main)
