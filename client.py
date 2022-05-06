import numpy as np
import glob
from PIL import Image
import os
import time
import imagezmq
import rospy, math
from geometry_msgs.msg import Point32
from sensor_msgs.msg import PointCloud
import pyrealsense2 as rs 
import cv2
from scipy.stats import skew
from threading import Thread
from multiprocessing import Process,Queue
import argparse

##function to convert depth to point clouds
def world_points(depth_image, x,y):
    z = depth_image.flatten() / 1000
    x = np.multiply(x, z)
    y = np.multiply(y, z)
    return np.reshape(np.dstack((x, y, z)), (480, 640, 3))

## function to publish data to ROS
def publisher(pc_queue):
    rospy.init_node("Segmenation_Data")
    pub = rospy.Publisher('segmentation/data', PointCloud, queue_size=1)
    while 1:
        ## if we have received the segmentation output from the server
        if (pc_queue.qsize()>0):
            st = time.time()               
            pc = pc_queue.get()
            pc = np.reshape(pc,(-1,3))

            segmentation_msgs=PointCloud()
            segmentation_msgs.header.stamp=rospy.Time.now()
            segmentation_msgs.header.frame_id="map"

            index=0
            for point in pc:
                segmentation_msgs.points.append(Point32())
                segmentation_msgs.points[index].x=point[0]
                segmentation_msgs.points[index].y=point[1]
                segmentation_msgs.points[index].z=point[2]
                index+=1
            pub.publish(segmentation_msgs)   
            print('publishing time: ', time.time()-st)
            del segmentation_msgs.points[:]
            
## remove artifact from depth map and prepare point clouds frames to send to server
def get_points(depth_image):
    if skew(depth_image.flatten())>2:
        depth_image[depth_image>np.percentile(depth_image.flatten(),95)] = 0
    depth_point = np.array(world_points(depth_image,x1,y1))
    ## we downscale the point cloud frames sent to server to reduce latency
    depth_scale = np.array(cv2.resize(depth_point,(320,240)))
    
    x = depth_point[:,:,0]
    y = depth_point[:,:,1]
    z = depth_point[:,:,2]

    x_scale = depth_scale[:,:,0]
    y_scale = depth_scale[:,:,1]
    z_scale = depth_scale[:,:,2]

    ## scale everything back to [0,1]
    x_scale = (x_scale-x_scale.min())/(x_scale.max()-x_scale.min()+1e-7)*255
    y_scale = (y_scale-y_scale.min())/(y_scale.max()-y_scale.min()+1e-7)*255
    z_scale = (z_scale-z_scale.min())/(z_scale.max()-z_scale.min()+1e-7)*255
    depth_scale = np.stack((x_scale,y_scale,z_scale),axis=-1).astype(np.uint8)
    return x,y,z,depth_scale

def get_raw_data():
    frames = pipeline.wait_for_frames()
    frames = align.process(frames)
    depth_frame = frames.get_depth_frame()
    color_frame = frames.get_color_frame()
    depth_image = np.asanyarray(depth_frame.get_data())
    color_image = np.asanyarray(color_frame.get_data()).astype(np.uint8)
    return color_image,depth_image

##function to create point cloud grid
def create_grid(depth_intrin):
    [height, width] = [480,640]
    nx = np.linspace(0, width - 1, width)
    ny = np.linspace(0, height - 1, height)
    u, v = np.meshgrid(nx, ny)
    x1 = (u.flatten() - depth_intrin.ppx) / depth_intrin.fx
    y1 = (v.flatten() - depth_intrin.ppy) / depth_intrin.fy
    return x1,y1

## function to merge the prediction with point cloud to send to ROS
def prepare_pc(x,y,z,label):
    ## we remove pixels around the image border the depth artifact might be there
    x = x[10:470,10:630]
    y = y[10:470,10:630]
    z = z[10:470,10:630]
    label = label[10:470,10:630]
    x = x[:,:,np.newaxis]
    y = y[:,:,np.newaxis]
    z = z[:,:,np.newaxis]
    label = label[:,:,np.newaxis]
    mask = (label==2)|(label==4)|(label==5) ##table,robot,backpack are also static obstacles
    label[mask]=0 ## static obstacle
    mask = (label==1)
    label[mask]=2   ## human
    mask = (label==6)
    label[mask]=1   ## free
    ## 3 is chair

    ## we also remove point clouds that are outside reliable range of RealSense
    ## uncertain class is also not sent to ROS
    ## we also removed points that is too high to avoid ceilling
    mask = (z<4)&(z>0.5)&(y>-1)&(label!=7)
    x = x[mask]
    z = z[mask]
    label = label[mask]    
    ## we remove y coordinates since the costmap does not need them
    point_cloud = np.stack((x,z,label),axis=-1)
    return point_cloud

def contact_server(color_image,depth_scale):
    send_imgs(color_image,'rgb')
    send_imgs(depth_scale,'depth')
    return 


## send images to server
def send_imgs(color_queue,depth_queue):
    sender = imagezmq.ImageSender(connect_to='tcp://'+remote_server+':5555')
    while 1:
        if (color_queue.qsize()>0)&(depth_queue.qsize()>0):
            rgb = color_queue.get()
            sender.send_jpg('rgb', rgb)
            
            depth = depth_queue.get()
            sender.send_jpg('depth', depth)

## receive images from server
def receive_imgs(label_queue,pc_queue,coor_queue):
    ## we opened a tcp port 5556
    image_hub = imagezmq.ImageHub(open_port='tcp://*:5556')
    while 1:
        st = time.time()
        _, png_buffer = image_hub.recv_jpg()
        image_hub.send_reply(b'OK')
        
        image = cv2.imdecode(np.frombuffer(png_buffer, dtype='uint8'), -1)
        ## the segmentation output has resolution (120,160)
        image = np.reshape(image,(120,160))
        ##resizing the label (we have to make things complicated since normal resize 
        # will cause inconsistencies around edges)
        image = image*30
        image = image.astype(np.uint8)
        image = np.array(cv2.resize(image,(640,480))).astype(np.uint8)
        mask = (image%30)!=0
        image[mask] = 210
        label = (image/30).astype('uint8')
        
        
        coor = coor_queue.get()
        coor = np.reshape(coor,(480,640,3))
        x,y,z = coor[:,:,0],coor[:,:,1],coor[:,:,2]
        
        label_queue.put(label.flatten())
        
        pc = prepare_pc(x,y,z,label)
        #np.random.shuffle(pc)
        ind = np.random.randint(0,len(pc),int(0.1*len(pc)))
        pc = pc[ind]
        print('len pc: ', len(pc))        
        pc_queue.put(pc.flatten())
        print('receiver time: ',time.time()-st)
        
## display on screen 
def display(rgb_queue,label_queue):
    while 1:
        if (rgb_queue.qsize()!=0)&(label_queue.qsize()!=0):
            rgb = rgb_queue.get()
            rgb = np.reshape(rgb,(480,640,3))
            label = label_queue.get()
            label = np.reshape(label,(480,640))
            label = np.array(label).astype(np.uint8)
            label = label[:,:,np.newaxis]            
            label = np.tile(label,(1,1,3)).astype('uint8')
            st = time.time()
            
            ## mapping between class label and color
            label[label[:,:,0]==0]=[255,0,0]
            label[label[:,:,0]==1]=[255,255,50]
            label[label[:,:,0]==2]=[28,255,0]    
            label[label[:,:,0]==3]=[0,255,255]
            label[label[:,:,0]==4]=[0,0,255]
            label[label[:,:,0]==5]=[127,0,255]
            label[label[:,:,0]==6]=[160,160,160]
            label[label[:,:,0]==7]=[0,0,0]
            print('display time: ', time.time()-st)
            #cv2.addWeighted(label, 0.3, rgb, 0.7 ,0, rgb)
            rgb = cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
            label = cv2.cvtColor(label,cv2.COLOR_RGB2BGR)
            
            cv2.imshow('label',label)
            cv2.waitKey(1)
            cv2.imshow('output',rgb)
            cv2.waitKey(1)

## compress image to JPEG to reduce latency 
def compress(img):
    return np.array(cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1]).astype('uint8')

if __name__ == '__main__':
    os.system('fuser -k 5556/tcp') ## kill any running 5556 port
    parser = argparse.ArgumentParser(description='client info')
    parser.add_argument('-server_dns', type=str ,help='dns displayed on aws')
    args = parser.parse_args()

    remote_server = str(args.server_dns)
    
    ##fire up RealSense
    pipeline = rs.pipeline()
    pipe_profile = pipeline.start()
    align = rs.align(rs.stream.color)   ##align depth frames and color frames

    ## wait for some iterations to warm up the camera
    for i in range(10):
            frames = pipeline.wait_for_frames()   
            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()

    depth_intrin = depth_frame.profile.as_video_stream_profile().intrinsics
    x1,y1 = create_grid(depth_intrin)

    ## create Queue and Multiprocessing Threads
    color_queue,depth_queue = Queue(),Queue()
    send_thread = Process(target=send_imgs,args=(color_queue,depth_queue))
    send_thread.daemon=True
    send_thread.start()
    
    pc_queue = Queue()
    publisher = Process(target=publisher,args=(pc_queue,))
    publisher.daemon=True
    publisher.start()

    rgb_queue,label_queue = Queue(),Queue()
    displayer = Process(target=display,args=(rgb_queue,label_queue))
    displayer.daemon=True
    displayer.start()

    coor_queue = Queue()
    receiver = Process(target=receive_imgs,args=(label_queue,pc_queue,coor_queue))
    receiver.daemon=True
    receiver.start()
    
    cnt = 0

    ##main thread
    while 1:
        st = time.time()

        ## get raw data from camera
        color_image,depth_image = get_raw_data()
        x,y,z,depth_scale = get_points(depth_image)
        coor = np.stack((x,y,z),axis=-1)  
        
        if (color_queue.qsize()==0) & (depth_queue.qsize()==0):
            color_queue.put(compress(color_image).flatten())
            depth_queue.put(compress(depth_scale).flatten())
            coor_queue.put(coor.flatten()) 
            rgb_queue.put(color_image.flatten()) 
        
        #if (rgb_queue.qsize()==0) & (coor_queue.qsize()==0):
            

        print('total time: ',time.time()-st)
        
        

	
