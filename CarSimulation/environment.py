import random
import time
import math
from gym import spaces
from tensorflow.keras.models import load_model
import numpy as np
import cv2
import gym
import carla
import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard

class ModifiedTensorBoard(TensorBoard):
    def __init__(self, log_dir, **kwargs):
        super().__init__(log_dir=log_dir, **kwargs)
        self.step = 1
        self.writer = tf.summary.create_file_writer(self.log_dir)

    def on_epoch_end(self, epoch, logs=None):
        self.update_stats(**logs)

    def on_batch_end(self, batch, logs=None):
        pass

    def on_train_end(self, _):
        pass

    def update_stats(self, **stats):
        with self.writer.as_default():
            for key, value in stats.items():
                tf.summary.scalar(key, value, step=self.step)
            self.writer.flush()



seconds_per_episode = 20
n_channels = 3
height = 240
width = 320
spin_angle = 15
height_required_portion = 0.5
width_required_portion = 0.9
show_preview = True


class CarEnv(gym.Env):
    show_cam = show_preview
    steer_amt = 1.0
    im_width = width
    im_height = height
    front_camera = None
    camera_pos_z = 1.8  # camera height
    camera_pos_x = 1.6  # camera distance from the vehicle
    preferred_speed = 40  # preferred speed
    speed_threshold = 5  # speed threshold

    def __init__(self):
        super(CarEnv, self).__init__()
        # Define action and observation space
        self.action_space = spaces.MultiDiscrete([9])
        self.height_from = int(height * (1 - height_required_portion))
        self.width_from = int((width - width * width_required_portion) / 2)
        self.width_to = self.width_from + int(width_required_portion * width)
        self.new_height = height - self.height_from
        self.new_width = self.width_to - self.width_from
        self.image_for_cnn = None
        self.observation_space = spaces.Box(low=0.0, high=1.0, shape=(7, 18, 8), dtype=np.float32)
        self.client = carla.Client("localhost", 2000)
        self.client.set_timeout(4.0)
        self.world = self.client.get_world()
        self.settings = self.world.get_settings()
        self.settings.no_rendering_mode = not self.show_cam
        self.world.apply_settings(self.settings)
        self.blueprint_library = self.world.get_blueprint_library()
        self.model_3 = self.blueprint_library.filter("vehicle.dodge.charger_2020")[0]
        self.cnn_model = load_model('D:\\Work\\Carla\\Model\\model_saved_from_CNN.h5', compile=False)
        self.cnn_model.compile()
        if self.show_cam:
            self.spectator = self.world.get_spectator()

    def cleanup(self):
        for sensor in self.world.get_actors().filter('*sensor*'):
            sensor.destroy()
        for actor in self.world.get_actors().filter('*vehicle*'):
            actor.destroy()
        cv2.destroyAllWindows()

    def maintain_speed(self, s):
        """
        Simple function to maintain the desired speed.
        """
        if s >= self.preferred_speed:
            return 0
        elif s < self.preferred_speed - self.speed_threshold:
            return 0.7  # Think of it as % of "full gas"
        else:
            return 0.3  # Tweak this if the car is way over or under preferred speed

    def apply_cnn(self, im):
        img = np.float32(im)
        img = img / 255
        img = np.expand_dims(img, axis=0)
        cnn_applied = self.cnn_model([img, 0], training=False)
        cnn_applied = np.squeeze(cnn_applied)
        return cnn_applied

    def step(self, action):
        trans = self.vehicle.get_transform()
        if self.show_cam:
            self.spectator.set_transform(
                carla.Transform(trans.location + carla.Location(z=20), carla.Rotation(yaw=-180, pitch=-90)))

        self.step_counter += 1
        steer = action[0]

        # Map steering actions
        if steer == 0:
            steer = -0.9
        elif steer == 1:
            steer = -0.25
        elif steer == 2:
            steer = -0.1
        elif steer == 3:
            steer = -0.05
        elif steer == 4:
            steer = 0.0
        elif steer == 5:
            steer = 0.05
        elif steer == 6:
            steer = 0.1
        elif steer == 7:
            steer = 0.25
        elif steer == 8:
            steer = 0.9

        if self.step_counter % 50 == 0:
            print('Steer input from model:', steer)

        v = self.vehicle.get_velocity()
        kmh = int(3.6 * math.sqrt(v.x ** 2 + v.y ** 2 + v.z ** 2))
        estimated_throttle = self.maintain_speed(kmh)
        self.vehicle.apply_control(carla.VehicleControl(throttle=estimated_throttle, steer=steer, brake=0.0))

        distance_travelled = self.initial_location.distance(self.vehicle.get_location())

        if self.show_cam:
            cv2.imshow('Camera View', self.front_camera)
            cv2.waitKey(1)

        lock_duration = 0
        if not self.steering_lock:
            if steer < -0.6 or steer > 0.6:
                self.steering_lock = True
                self.steering_lock_start = time.time()
        else:
            if steer < -0.6 or steer > 0.6:
                lock_duration = time.time() - self.steering_lock_start

        reward = 0
        done = False
        if len(self.collision_hist) != 0:
            done = True
            reward -= 300
            self.cleanup()
        if len(self.lane_invade_hist) != 0:
            done = True
            reward -= 300
            self.cleanup()
        if lock_duration > 3:
            reward -= 150
            done = True
            self.cleanup()
        elif lock_duration > 1:
            reward -= 20
        if distance_travelled < 30:
            reward -= 1
        elif distance_travelled < 50:
            reward += 1
        else:
            reward += 2
        if self.episode_start + seconds_per_episode < time.time():
            done = True
            self.cleanup()
        self.image_for_cnn = self.apply_cnn(self.front_camera[self.height_from:, self.width_from:self.width_to])

        return self.image_for_cnn, reward, done, {}

    def reset(self):
        self.collision_hist = []
        self.lane_invade_hist = []
        self.actor_list = []
        self.transform = random.choice(self.world.get_map().get_spawn_points())
        self.vehicle = None
        while self.vehicle is None:
            try:
                self.vehicle = self.world.spawn_actor(self.model_3, self.transform)
            except:
                pass
        self.actor_list.append(self.vehicle)
        self.initial_location = self.vehicle.get_location()
        self.sem_cam = self.blueprint_library.find('sensor.camera.semantic_segmentation')
        self.sem_cam.set_attribute("image_size_x", f"{self.im_width}")
        self.sem_cam.set_attribute("image_size_y", f"{self.im_height}")
        self.sem_cam.set_attribute("fov", f"90")
        camera_init_trans = carla.Transform(carla.Location(z=self.camera_pos_z, x=self.camera_pos_x))
        self.sensor = self.world.spawn_actor(self.sem_cam, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.sensor)
        self.sensor.listen(lambda data: self.process_img(data))
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        time.sleep(2)
        angle_adj = random.randrange(-spin_angle, spin_angle, 1)
        trans = self.vehicle.get_transform()
        trans.rotation.yaw += angle_adj
        self.vehicle.set_transform(trans)
        colsensor = self.blueprint_library.find("sensor.other.collision")
        self.colsensor = self.world.spawn_actor(colsensor, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.colsensor)
        self.colsensor.listen(lambda event: self.collision_data(event))
        lanesensor = self.blueprint_library.find("sensor.other.lane_invasion")
        self.lanesensor = self.world.spawn_actor(lanesensor, camera_init_trans, attach_to=self.vehicle)
        self.actor_list.append(self.lanesensor)
        self.lanesensor.listen(lambda event: self.lane_data(event))
        while self.front_camera is None:
            time.sleep(0.01)
        self.episode_start = time.time()
        self.steering_lock = False
        self.steering_lock_start = None
        self.step_counter = 0
        self.vehicle.apply_control(carla.VehicleControl(throttle=0.0, brake=0.0))
        self.image_for_cnn = self.apply_cnn(self.front_camera[self.height_from:, self.width_from:self.width_to])
        return self.image_for_cnn

    def process_img(self, image):
        image.convert(carla.ColorConverter.CityScapesPalette)
        i = np.array(image.raw_data)
        i = i.reshape((self.im_height, self.im_width, 4))[:, :, :3]  # Ignore the 4th alpha channel
        self.front_camera = i

    def collision_data(self, event):
        self.collision_hist.append(event)

    def lane_data(self, event):
        self.lane_invade_hist.append(event)

