import cv2
import h5py
import numpy as np

from src.adapters.hdf5_adapter import HDF5Adapter
from src.core.factory import ReaderFactory
from src.core.interface import AdapterConfig


def _write_test_video(path, bgr_frames, fps=10):
    height, width = bgr_frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    assert writer.isOpened()
    for frame in bgr_frames:
        writer.write(frame)
    writer.release()


def test_hdf5_blank_images_fall_back_to_configured_video(tmp_path):
    h5_path = tmp_path / "episode_000000.hdf5"
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_path = video_dir / "cam_head.avi"

    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("action", data=np.zeros((2, 3), dtype=np.float32))
        image_group = h5_file.create_group("observations/images")
        image_group.create_dataset(
            "cam_head",
            data=np.zeros((2, 8, 8, 3), dtype=np.uint8),
        )

    red_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    red_bgr[..., 2] = 255
    green_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    green_bgr[..., 1] = 255
    _write_test_video(video_path, [red_bgr, green_bgr])

    reader = HDF5Adapter(
        AdapterConfig(
            length_reference_key="action",
            image_keys_map={"cam_head": "observations/images/cam_head"},
            extra_options={
                "image_source": "auto",
                "videos": {"cam_head": "videos/cam_head.avi"},
            },
        )
    )

    try:
        assert reader.load(str(h5_path))
        frame = reader.get_frame(0)
    finally:
        reader.close()

    image = frame.images["cam_head"]
    assert image[..., 0].mean() > 200
    assert image[..., 1].mean() < 50
    assert image[..., 2].mean() < 50


def test_hdf5_video_source_can_define_video_only_camera(tmp_path):
    h5_path = tmp_path / "episode_000000.hdf5"
    video_dir = tmp_path / "videos"
    video_dir.mkdir()
    video_path = video_dir / "cam_only.avi"

    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("action", data=np.zeros((1, 3), dtype=np.float32))

    blue_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    blue_bgr[..., 0] = 255
    _write_test_video(video_path, [blue_bgr])

    reader = HDF5Adapter(
        AdapterConfig(
            length_reference_key="action",
            extra_options={
                "image_source": "video",
                "videos": {"cam_only": "videos/cam_only.avi"},
            },
        )
    )

    try:
        assert reader.load(str(h5_path))
        assert reader.get_all_sensors() == ["cam_only"]
        frame = reader.get_frame(0)
    finally:
        reader.close()

    image = frame.images["cam_only"]
    assert image[..., 0].mean() < 50
    assert image[..., 1].mean() < 50
    assert image[..., 2].mean() > 200


def test_hdf5_embedded_video_dataset_can_be_used_as_image_source(tmp_path):
    h5_path = tmp_path / "episode_000000.hdf5"
    video_path = tmp_path / "cam_head.avi"

    red_bgr = np.zeros((8, 8, 3), dtype=np.uint8)
    red_bgr[..., 2] = 255
    _write_test_video(video_path, [red_bgr])
    video_bytes = np.frombuffer(video_path.read_bytes(), dtype=np.uint8)

    with h5py.File(h5_path, "w") as h5_file:
        h5_file.create_dataset("action", data=np.zeros((1, 3), dtype=np.float32))
        camera_group = h5_file.create_group("observations/camera/rgb/head")
        camera_group.create_dataset(
            "images",
            data=np.zeros((0,), dtype=h5py.vlen_dtype(np.dtype("uint8"))),
        )
        camera_group.create_dataset("video", data=video_bytes)

    reader = HDF5Adapter(
        AdapterConfig(
            length_reference_key="action",
            image_keys_map={"cam_head": "observations/camera/rgb/head/images"},
            extra_options={
                "image_source": "auto",
                "videos": {"cam_head": "observations/camera/rgb/head/video"},
            },
        )
    )

    try:
        assert reader.load(str(h5_path))
        frame = reader.get_frame(0)
        temp_files = list(reader.video_temp_files.values())
        assert temp_files and temp_files[0].exists()
    finally:
        reader.close()

    assert not temp_files[0].exists()
    image = frame.images["cam_head"]
    assert image[..., 0].mean() > 200
    assert image[..., 1].mean() < 50
    assert image[..., 2].mean() < 50


def test_factory_passes_hdf5_video_options_from_rule(monkeypatch):
    monkeypatch.setattr(
        ReaderFactory,
        "_rules_cache",
        {
            "RobotH5": {
                "base_type": "HDF5",
                "length_reference_key": "action",
                "image_source": "auto",
                "cameras": {"cam_head": "observations/images/cam_head"},
                "videos": {"cam_head": "observations/camera/rgb/head/video"},
            }
        },
    )

    reader = ReaderFactory.get_reader("episode.hdf5", rule_name="RobotH5")

    assert isinstance(reader, HDF5Adapter)
    assert reader.camera_map == {"cam_head": "observations/images/cam_head"}
    assert reader.video_map == {"cam_head": "observations/camera/rgb/head/video"}
    assert reader.image_source == "auto"
