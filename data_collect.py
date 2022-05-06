import numpy as np                            
from PIL import Image
import pyrealsense2 as rs  
import os
import time
from scipy.stats import skew

## for converting depth into point clouds
def world_points(depth_image, camera_intrinsics):
    
    [height, width] = depth_image.shape
    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x = (u.flatten() - camera_intrinsics.ppx) / camera_intrinsics.fx
    y = (v.flatten() - camera_intrinsics.ppy) / camera_intrinsics.fy
    
    z = depth_image.flatten() / 1000
    x = np.multiply(x, z)
    y = np.multiply(y, z)
    
    return np.reshape(np.dstack((x, y, z)), (height, width, 3))

##fire up the RealSense
pipeline = rs.pipeline()
pipe_profile = pipeline.start()
align = rs.align(rs.stream.color)
pc = rs.pointcloud()

frame_num=0

## warming up the camera
for i in range(7):
      frames = pipeline.wait_for_frames()   
      frames = align.process(frames)
      depth_frame = frames.get_depth_frame()

while 1:  
  frames = pipeline.wait_for_frames()
  frames = align.process(frames)
  depth_frame = frames.get_depth_frame()
  color_frame = frames.get_color_frame()
  depth_image = np.asanyarray(depth_frame.get_data())
  color_image = np.asanyarray(color_frame.get_data())
  depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
  
  ## depth image is usally skewed when there is depth artifacts when pointing at glass surfaces/window
  if skew(depth_image.flatten())>5:
        depth_image[depth_image>np.percentile(depth_image.flatten(),95)] = 0
  depth_point = np.array(world_points(depth_image,depth_intrin))

  st = time.time()
  x = depth_point[:,:,0]
  y = depth_point[:,:,1]
  z = depth_point[:,:,2]

  ## scale back the point cloud images to integer from [0,255]
  x = (x-x.min())/(x.max()-x.min())*255
  y = (y-y.min())/(y.max()-y.min())*255
  z = (z-z.min())/(z.max()-z.min())*255
  
  path = '/home/frauas/segmentation/data/custom/'
  path_rgb = os.path.join(path+"rgb","{}.jpg".format(frame_num))
  path_x = os.path.join(path+"x","{}.jpg".format(frame_num))
  path_y = os.path.join(path+"y","{}.jpg".format(frame_num))
  path_z = os.path.join(path+"z","{}.jpg".format(frame_num))
  
  ## we just save 1 every 20 frames
  if frame_num%20==0:
        Image.fromarray(color_image).save(path_rgb)
        depth_point = np.array(depth_point)
        Image.fromarray((x).astype(np.uint8)).save(path_x)
        Image.fromarray((y).astype(np.uint8)).save(path_y)
        Image.fromarray((z).astype(np.uint8)).save(path_z)
  
  frame_num+=1
  
  