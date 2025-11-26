from qtpy.QtWidgets import QLabel


class ScaledImageLabel(QLabel):
    def __init__(self, parent=None, rescale_callback=None):
        """Initialize the resizable label.

        Args:
            parent: Parent widget
            rescale_callback: Callback function to execute on resize events
        """
        super().__init__(parent)
        self._rescale_callback = rescale_callback

    def resizeEvent(self, event):
        """Handle resize events and trigger rescale callback.

        Args:
            event: QResizeEvent object
        """
        if self._rescale_callback:
            self._rescale_callback()
        super().resizeEvent(event)

