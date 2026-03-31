from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence


class EpisodeDatasetReader:
    """Reader for the current dataset layout.

    Expected layout:
    - <dataset_root>/action/0000.json
    - <dataset_root>/videos/chunk-000/<camera_name>/episode_0000.mp4
    """

    def __init__(
        self,
        dataset_root: str | Path,
        action_dir_name: str = "action",
        video_dir_name: str = "videos",
        cache_action_data: bool = True,
    ):
        self.dataset_root = Path(dataset_root).expanduser().resolve()
        self.action_root = self.dataset_root / action_dir_name
        self.video_root = self.dataset_root / video_dir_name
        self.cache_action_data = cache_action_data

        self._action_file_map = self._build_action_file_map()
        self._video_file_map = self._build_video_file_map()
        self._action_cache: Dict[int, List[Dict[str, Any]]] = {}

    def _build_action_file_map(self) -> Dict[int, Path]:
        action_file_map: Dict[int, Path] = {}
        if not self.action_root.exists():
            return action_file_map

        for json_path in sorted(self.action_root.glob("*.json")):
            try:
                episode_num = int(json_path.stem)
            except ValueError:
                continue
            action_file_map[episode_num] = json_path
        return action_file_map

    def _build_video_file_map(self) -> Dict[int, Dict[str, Path]]:
        video_file_map: Dict[int, Dict[str, Path]] = {}
        if not self.video_root.exists():
            return video_file_map

        for video_path in sorted(self.video_root.rglob("episode_*.mp4")):
            try:
                episode_num = int(video_path.stem.split("_")[-1])
            except ValueError:
                continue

            camera_name = video_path.parent.name
            video_file_map.setdefault(episode_num, {})[camera_name] = video_path
        return video_file_map

    def has_episode(self, episode_num: int) -> bool:
        episode_num = int(episode_num)
        return episode_num in self._action_file_map or episode_num in self._video_file_map

    def available_episodes(self) -> List[int]:
        return sorted(set(self._action_file_map.keys()) | set(self._video_file_map.keys()))

    def existing_episodes(self, episode_nums: Sequence[int]) -> List[int]:
        return [int(ep) for ep in episode_nums if self.has_episode(int(ep))]

    def missing_episodes(self, episode_nums: Sequence[int]) -> List[int]:
        return [int(ep) for ep in episode_nums if not self.has_episode(int(ep))]

    def _load_action_data(self, episode_num: int) -> Optional[List[Dict[str, Any]]]:
        episode_num = int(episode_num)
        action_path = self._action_file_map.get(episode_num)
        if action_path is None:
            return None

        if episode_num in self._action_cache:
            return self._action_cache[episode_num]

        with action_path.open("r", encoding="utf-8") as f:
            action_data = json.load(f)

        if self.cache_action_data:
            self._action_cache[episode_num] = action_data
        return action_data

    def _collect_video_paths(self, episode_num: int) -> Dict[str, Path]:
        return dict(self._video_file_map.get(int(episode_num), {}))

    def _extract_video_frames(
        self,
        video_paths: Dict[str, Path],
        frame_indices: Sequence[int],
        load_images: bool,
    ) -> Dict[str, Any]:
        if not video_paths:
            return {}

        if not load_images:
            return {
                camera_name: {
                    "video_path": str(video_path),
                    "frame_indices": list(frame_indices),
                }
                for camera_name, video_path in video_paths.items()
            }

        try:
            import cv2
        except ImportError as e:
            raise ImportError("opencv-python is required when load_images=True") from e

        frames_by_camera: Dict[str, List[Any]] = {}
        for camera_name, video_path in video_paths.items():
            capture = cv2.VideoCapture(str(video_path))
            if not capture.isOpened():
                raise RuntimeError(f"failed to open video: {video_path}")

            frames: List[Any] = []
            try:
                for frame_idx in frame_indices:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_idx))
                    ok, frame_bgr = capture.read()
                    if not ok:
                        frames.append(None)
                        continue
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    frames.append(frame_rgb)
            finally:
                capture.release()

            frames_by_camera[camera_name] = frames

        return frames_by_camera

    def _extract_joint_positions(self, action_data: Optional[List[Dict[str, Any]]]) -> Optional[List[Any]]:
        if action_data is None:
            return None
        return [
            frame.get("robot", {}).get("joint_positions")
            for frame in action_data
        ]

    def _extract_indices(self, action_data: Optional[List[Dict[str, Any]]]) -> Optional[List[Any]]:
        if action_data is None:
            return None
        return [frame.get("index") for frame in action_data]

    def get_episode_data(
        self,
        episode_num: int,
        include: Sequence[str] = ("image", "joint", "index"),
        load_images: bool = False,
    ) -> Dict[str, Any]:
        episode_num = int(episode_num)
        include_set = {item.lower() for item in include}

        result: Dict[str, Any] = {
            "episode_num": episode_num,
            "exists": self.has_episode(episode_num),
            "action_path": str(self._action_file_map[episode_num]) if episode_num in self._action_file_map else None,
            "video_paths": {
                camera_name: str(video_path)
                for camera_name, video_path in self._video_file_map.get(episode_num, {}).items()
            },
        }

        if not result["exists"]:
            return result

        action_data = self._load_action_data(episode_num) if ("joint" in include_set or "index" in include_set) else None
        if action_data is None and "image" in include_set:
            action_data = self._load_action_data(episode_num)

        if "image" in include_set:
            frame_indices = self._extract_indices(action_data) or []
            video_paths = self._collect_video_paths(episode_num)
            result["image"] = self._extract_video_frames(
                video_paths=video_paths,
                frame_indices=frame_indices,
                load_images=load_images,
            )

        if "joint" in include_set:
            result["joint"] = self._extract_joint_positions(action_data)

        if "index" in include_set:
            result["index"] = self._extract_indices(action_data)

        return result

    def get_many(
        self,
        episode_nums: Sequence[int],
        include: Sequence[str] = ("image", "joint", "index"),
        load_images: bool = False,
        verbose: bool = True,
    ) -> Dict[int, Dict[str, Any]]:
        results: Dict[int, Dict[str, Any]] = {}

        for episode_num in episode_nums:
            episode_num = int(episode_num)
            episode_data = self.get_episode_data(
                episode_num=episode_num,
                include=include,
                load_images=load_images,
            )

            if not episode_data["exists"]:
                if verbose:
                    print(f"[EpisodeDatasetReader] episode {episode_num} does not exist")
                continue

            results[episode_num] = episode_data
            if verbose:
                self.print_episode_summary(episode_data)

        return results

    def print_episode_summary(self, episode_data: Dict[str, Any], preview_count: int = 3) -> None:
        print(f"\n[Episode {episode_data['episode_num']}]")
        print(f"action_path: {episode_data.get('action_path')}")
        print(f"video_paths: {episode_data.get('video_paths')}")

        if "image" in episode_data:
            image_data = episode_data["image"]
            print("image:")
            if not image_data:
                print("  no video/image found")
            else:
                for camera_name, values in image_data.items():
                    if isinstance(values, dict):
                        preview = values
                    else:
                        preview_items = values[:preview_count]
                        preview = []
                        for item in preview_items:
                            if item is None:
                                preview.append("None")
                            elif hasattr(item, "shape"):
                                preview.append(f"ndarray(shape={tuple(item.shape)})")
                            else:
                                preview.append(str(item))
                    print(f"  {camera_name}: count={len(values)}, preview={preview}")

        if "joint" in episode_data:
            joint_data = episode_data["joint"]
            if joint_data is None:
                print("joint: no action json found")
            else:
                print(f"joint: count={len(joint_data)}, preview={joint_data[:preview_count]}")

        if "index" in episode_data:
            index_data = episode_data["index"]
            if index_data is None:
                print("index: no action json found")
            else:
                print(f"index: count={len(index_data)}, preview={index_data[:preview_count]}")


def load_episode_data(
    dataset_root: str | Path,
    episode_nums: Sequence[int],
    include: Sequence[str] = ("image", "joint", "index"),
    load_images: bool = False,
    cache_action_data: bool = True,
    verbose: bool = True,
) -> Dict[int, Dict[str, Any]]:
    reader = EpisodeDatasetReader(
        dataset_root=dataset_root,
        cache_action_data=cache_action_data,
    )
    return reader.get_many(
        episode_nums=episode_nums,
        include=include,
        load_images=load_images,
        verbose=verbose,
    )
