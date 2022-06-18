"""Wrapper for recording videos."""
import json
import os
from typing import Callable, List

import gym
from gym import error, logger

RECORDING_EXT = "mp4"


def capped_cubic_video_schedule(episode_id: int) -> bool:
    """The default episode trigger.

    This function will trigger recordings at the episode indices 0, 1, 4, 8, 27, ..., :math:`k^3`, ..., 729, 1000, 2000, 3000, ...

    Args:
        episode_id: The episode number

    Returns:
        If to apply a video schedule number
    """
    if episode_id < 1000:
        return int(round(episode_id ** (1.0 / 3))) ** 3 == episode_id
    else:
        return episode_id % 1000 == 0


class RecordVideo(gym.Wrapper):
    """This wrapper records videos of rollouts with MoviePy library.

    The environment provided has to be initialized with ``render_mode`` as 'single_rgb_array' or 'rgb_array' in order to
    record the rendered output.

    Usually, you only want to record episodes intermittently, say every hundredth episode.
    To do this, you can specify **either** ``episode_trigger`` **or** ``step_trigger`` (not both).
    They should be functions returning a boolean that indicates whether a recording should be started at the
    current episode or step, respectively.
    If neither :attr:`episode_trigger` nor ``step_trigger`` is passed, a default ``episode_trigger`` will be employed.
    By default, the recording will be stopped once a `done` signal has been emitted by the environment. However, you can
    also create recordings of fixed length (possibly spanning several episodes) by passing a strictly positive value for
    ``video_length``.
    """

    def __init__(
        self,
        env: gym.Env,
        video_folder: str,
        episode_trigger: Callable[[int], bool] = None,
        step_trigger: Callable[[int], bool] = None,
        video_length: int = 0,
        name_prefix: str = "rl-video",
    ):
        """Wrapper records videos of rollouts.

        Args:
            env: The environment that will be wrapped
            video_folder (str): The folder where the recordings will be stored
            episode_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this episode
            step_trigger: Function that accepts an integer and returns ``True`` iff a recording should be started at this step
            video_length (int): The length of recorded episodes. If 0, entire episodes are recorded.
                Otherwise, snippets of the specified length are captured
            name_prefix (str): Will be prepended to the filename of the recordings
        """
        super().__init__(env)

        self.enabled = True

        if "rgb_array" == env.render_mode or "single_rgb_array" == env.render_mode:
            try:
                # Check library availability
                from moviepy.tools import extensions_dict
            except ImportError:
                logger.warn(
                    "Disabling video recorder because MoviePy is not installed, run `pip install moviepy`."
                )  # FIXME is it ok, or an exception is needed?
            self.enabled = True
            self.codec = extensions_dict[RECORDING_EXT]["codec"][0]
        else:
            if "ansi" == env.render_mode:
                logger.deprecation(
                    "Video recorder support for for ansi rendering mode is deprecated."
                )
            else:
                logger.info(
                    f"Disabling video recorder because {env} has not been initialized with a compatible render mode"
                    "between 'single_rgb_array' and 'rgb_array'."
                )
            # Disable since the environment has not been initialized with a compatible `render_mode`
            self.enabled = False

        if episode_trigger is None and step_trigger is None:
            episode_trigger = capped_cubic_video_schedule

        trigger_count = sum(x is not None for x in [episode_trigger, step_trigger])
        assert trigger_count == 1, "Must specify exactly one trigger"

        self.async_env = env.metadata.get("semantics.async")
        self.output_frames_per_sec = env.metadata.get("render_fps", 30)

        # FIXME do you confirm to delete this?
        # # backward-compatibility mode:
        # self.backward_compatible_output_frames_per_sec = env.metadata.get(
        #     "video.output_frames_per_second", self.output_frames_per_sec
        # )
        # if self.output_frames_per_sec != self.backward_compatible_output_frames_per_sec:
        #     logger.deprecation(
        #         '`env.metadata["video.output_frames_per_second"] is marked as deprecated and will be replaced '
        #         'with `env.metadata["render_fps"]` see https://github.com/openai/gym/pull/2654 for more details'
        #     )
        #     self.output_frames_per_sec = self.backward_compatible_output_frames_per_sec

        self.video_folder = os.path.abspath(video_folder)
        # Create output folder if needed
        if os.path.isdir(self.video_folder):
            logger.warn(
                f"Overwriting existing videos at {self.video_folder} folder "
                f"(try specifying a different `video_folder` for the `RecordVideo` wrapper if this is not desired)"
            )
        os.makedirs(self.video_folder, exist_ok=True)

        self.episode_trigger = episode_trigger
        self.step_trigger = step_trigger
        self.video_length = video_length
        self.name_prefix = name_prefix

        self.recording = False
        self.recorded_frames = 0
        self.frames = []
        self.is_vector_env = getattr(env, "is_vector_env", False)
        self.episode_id = 0
        self.step_id = 0

    def reset(self, **kwargs):
        """Reset the environment using kwargs and then starts recording if video enabled."""
        observations = super().reset(**kwargs)
        if self.recording:
            self._capture_frame()
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
        elif self._video_enabled():
            self.start_video_recorder()
        return observations

    def start_video_recorder(self):
        """Starts video recording."""
        self.close_video_recorder()

        video_name = f"{self.name_prefix}-step-{self.step_id}"
        if self.episode_trigger:
            video_name = f"{self.name_prefix}-episode-{self.episode_id}"
        base_path = os.path.join(self.video_folder, video_name)

        self._current_rec_filepath = base_path + "." + RECORDING_EXT
        self._current_rec_meta_filepath = base_path + ".meta.json"
        self._broken_recording = False

        # Dump metadata
        self._current_rec_metadata = {
            "step_id": self.step_id,
            "episode_id": self.episode_id,
            "content_type": "video/" + RECORDING_EXT,
            "codec": self.codec,
            "empty": False,
            "broken": False,
        }
        self._write_metadata()

        logger.info(f"Starting new video recording {self._current_rec_filepath}")

        # First capture
        self.recording = True
        self._capture_frame()
        self.recorded_frames = 1

    def _video_enabled(self):
        if self.enabled:
            if self.step_trigger:
                return self.step_trigger(self.step_id)
            else:
                return self.episode_trigger(self.episode_id)

    def step(self, action):
        """Steps through the environment using action, recording observations if :attr:`self.recording`."""
        observations, rewards, dones, infos = super().step(action)

        # Increment steps and episodes
        self.step_id += 1
        if not self.is_vector_env:
            if dones:
                self.episode_id += 1
        elif dones[0]:
            self.episode_id += 1

        if self.recording:
            self._capture_frame()  # FIXME even if broken the frame counter will continue to be increased. Is the intended behaviour?
            self.recorded_frames += 1
            if self.video_length > 0:
                if self.recorded_frames > self.video_length:
                    self.close_video_recorder()
            else:
                if not self.is_vector_env:
                    if dones:
                        self.close_video_recorder()
                elif dones[0]:
                    self.close_video_recorder()

        elif self._video_enabled():
            self.start_video_recorder()

        return observations, rewards, dones, infos

    def _capture_frame(self):
        """Append the current rendering of the environment to the stored frames."""
        if not self._broken_recording:
            frame = self.env.render()

            if isinstance(frame, List):
                frame = frame[-1]

            if frame is None:
                if self.async_env:
                    return
                else:
                    # There must be an error in the environment
                    logger.warn(
                        "Env returned None on `render()`. Disabling further rendering for video recorder by marking as "
                        f"disabled: path={self._current_rec_filepath} metadata_path={self._current_rec_meta_filepath}"
                    )
                    self._broken_recording = True
            else:
                self.frames.append(frame)

    def close_video_recorder(self):
        """Closes the video recorder if currently recording."""
        if self.recording:
            if len(self.frames) > 0:
                from moviepy.video.io.ImageSequenceClip import ImageSequenceClip

                try:
                    # Save frames as a video file
                    clip = ImageSequenceClip(
                        self.frames, fps=self.output_frames_per_sec
                    )
                    clip.write_videofile(
                        self._current_rec_filepath, verbose=False, logger=None
                    )

                    logger.info(
                        "Saving recorded video: path=%s", self._current_rec_filepath
                    )
                except error.InvalidFrame as e:
                    logger.warn(
                        "Tried to pass invalid video frame, marking as broken: %s", e
                    )
                    self._broken_recording = True
            else:
                # Update metadata
                self._current_rec_metadata["empty"] = True

            # If broken, get rid of the output file, otherwise we'd leak it.
            if self._broken_recording:
                logger.info(
                    "Cleaning up paths for broken video recorder: path=%s",
                    self._current_rec_filepath,
                )

                # Might have crashed before even starting the output file, don't try to remove in that case.
                if os.path.exists(self._current_rec_filepath):
                    os.remove(self._current_rec_filepath)

                # Update metadata
                self._current_rec_metadata["broken"] = True
            self._write_metadata()
        # Reset variables
        self.frames = []
        self.recording = False
        self.recorded_frames = 0

    def _write_metadata(self):
        """Writes metadata to metadata path."""
        with open(self._current_rec_meta_filepath, "w") as f:
            json.dump(self._current_rec_metadata, f)

    def close(self):
        """Closes the wrapper then the video recorder."""
        super().close()
        self.close_video_recorder()

    def __del__(self):
        """Closes the video recorder."""
        self.close_video_recorder()
