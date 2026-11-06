from PIL import Image

from robustness import PERTURBATIONS, darken, brighten


def test_perturbations_return_rgb_images_of_same_dimensions():
    image = Image.new("RGB", (48, 32), color=(120, 120, 120))

    assert set(PERTURBATIONS) == {
        "blur_radius_2",
        "rotation_15_degrees",
        "brightness_50_percent",
        "brightness_150_percent",
    }
    for perturbation in PERTURBATIONS.values():
        result = perturbation(image)
        assert result.mode == "RGB"
        assert result.size == image.size


def test_lighting_perturbations_change_brightness_in_expected_direction():
    image = Image.new("RGB", (8, 8), color=(100, 100, 100))

    assert sum(darken(image).getpixel((0, 0))) < sum(image.getpixel((0, 0)))
    assert sum(brighten(image).getpixel((0, 0))) > sum(image.getpixel((0, 0)))
