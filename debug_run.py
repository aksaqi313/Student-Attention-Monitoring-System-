import sys, traceback, os

# Redirect stderr to capture mediapipe warnings
import io
_orig_stderr = sys.stderr
sys.stderr = io.StringIO()

try:
    from main import AttentionApp
    app = AttentionApp()
    app.after(1000, app.destroy)
    app.mainloop()
except Exception as e:
    sys.stderr = _orig_stderr
    print("=" * 60)
    print("FULL TRACEBACK:")
    traceback.print_exc()
    print("=" * 60)
    print("MSG:", str(e))
finally:
    sys.stderr = _orig_stderr
