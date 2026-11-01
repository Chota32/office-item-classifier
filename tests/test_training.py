import subprocess
import sys

def test_training():
    result = subprocess.run([sys.executable, "app.py", "train", "--epochs", "1"], capture_output=True, text=True)

    assert result.returncode == 0, result.stderr
    assert "Training loss:" in result.stdout
    assert "Validation loss:" in result.stdout

    print("Training test passed!")

if __name__ == "__main__":
    test_training()