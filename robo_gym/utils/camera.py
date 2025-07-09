#!/usr/bin/env python3
import numpy as np
import cv2
import base64

class RoboGymCamera:
    def __init__(self, name, image_shape, image_mode, context_size, num_cameras):
        self.name = name
        self.image_shape = image_shape
        self.image_mode = image_mode
        self.context_size = context_size
        self.num_cameras = num_cameras

        self.observation_space = self.get_camera_observation_shape()

    def process_camera_images(self, string_params):
        """Process camera images from string_params into proper format."""

        image_count = int(string_params.get("image_count", "0"))

        if image_count == 0:
            if self.image_mode == 'temporal':
                stacked_shape = (self.image_shape[0], self.image_shape[1],
                            self.image_shape[2] * self.context_size)
                return np.zeros(stacked_shape, dtype=np.uint8)
            else:
                return np.zeros(self.image_shape, dtype=np.uint8)

        images = []
        for i in range(image_count):
            image_key = f"camera_image_{i}"
            if image_key in string_params:
                try:
                    image_string = string_params[image_key]
                    image_bytes = base64.b64decode(image_string.encode('utf-8'))
                    nparr = np.frombuffer(image_bytes, np.uint8)
                    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

                    if image.shape != self.image_shape:
                        image = cv2.resize(image, (self.image_shape[1], self.image_shape[0]))

                    images.append(image)

                except Exception as e:
                    print(f"Warning: Failed to decode camera_image_{i}: {e}")
                    images.append(np.zeros(self.image_shape, dtype=np.uint8))

        if self.image_mode == 'temporal':
            while len(images) < self.context_size:
                images.insert(0, np.zeros(self.image_shape, dtype=np.uint8))

            images = images[-self.context_size:]

            stacked_images = np.concatenate(images, axis=2)
            return stacked_images.astype(np.uint8)

        elif self.image_mode == 'single':
            if images:
                return images[-1].astype(np.uint8)
            else:
                return np.zeros(self.image_shape, dtype=np.uint8)

        else:
            raise ValueError(f"Unsupported image_mode: {self.image_mode}")

    def get_camera_observation_shape(self):
        """Calculate camera observation shape based on configuration."""

        base_height, base_width, base_channels = self.image_shape

        if self.image_mode == 'single':
            return (base_height, base_width, base_channels)

        elif self.image_mode == 'temporal':
            stacked_channels = base_channels * self.context_size
            return (base_height, base_width, stacked_channels)

        elif self.image_mode == 'multi_camera':
            return (base_height, base_width * self.num_cameras, base_channels)

        elif self.image_mode == 'hybrid':

            temporal_channels = base_channels * self.context_size
            total_channels = temporal_channels * self.num_cameras
            return (base_height, base_width, total_channels)

        else:
            raise ValueError(f"Unknown image_mode: {self.image_mode}")
