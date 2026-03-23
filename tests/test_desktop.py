"""Tests for the desktop launcher module."""

import socket


class TestFindFreePort:
    def test_returns_valid_port(self):
        """find_free_port returns an integer in valid range."""
        from desktop import find_free_port
        port = find_free_port()
        assert isinstance(port, int)
        assert 1024 <= port <= 65535

    def test_port_is_available(self):
        """The returned port should be bindable."""
        from desktop import find_free_port
        port = find_free_port()
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.bind(("localhost", port))


class TestWaitForServer:
    def test_succeeds_on_open_port(self):
        """wait_for_server returns True when a server is listening."""
        from desktop import wait_for_server
        # Start a temporary server
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.bind(("localhost", 0))
        port = server.getsockname()[1]
        server.listen(1)
        try:
            assert wait_for_server(port, timeout=3) is True
        finally:
            server.close()

    def test_fails_on_closed_port(self):
        """wait_for_server returns False when nothing is listening."""
        from desktop import find_free_port, wait_for_server
        port = find_free_port()
        assert wait_for_server(port, timeout=2) is False


class TestDesktopAppImport:
    def test_main_is_callable(self):
        """desktop.main should be importable and callable."""
        from desktop import main
        assert callable(main)


class TestFormatSize:
    def test_bytes(self):
        from app import _format_size
        assert "B" in _format_size(500)

    def test_megabytes(self):
        from app import _format_size
        result = _format_size(5 * 1024 * 1024)
        assert "MB" in result

    def test_gigabytes(self):
        from app import _format_size
        result = _format_size(7 * 1024 * 1024 * 1024)
        assert "GB" in result

    def test_zero(self):
        from app import _format_size
        assert "0.0 B" == _format_size(0)
