import math
import os

import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Circle
import networkx as nx
import numpy as np
import torch
from PIL import Image
from astropy.stats import sigma_clipped_stats
from photutils import find_peaks
from Patch import Patch
from SeedStack import SeedStack
import time

from roads import HourGlass as nt
import roads.patch.bifurcations_toolbox_roads as tb

class road_tracking():

    root_dir='/data/MassachusettsRoads/TestSet/test_input/'
    model_dir = '/results/weights/'
    save_topology_root = '/results/topology/'
    checkname = '/results/weights/weights.pth'

    visited_path_mask = None
    visited_path = [] 
    crop_size = 64 
    step_size = 20 
    offset = 0 
    confidence_th = 0.05 
    auto_seed_confidence_th = 0.15
    connection_th = 10
    close_visited_th = 1 

    visualize_step = False 
    use_line_conn = True 
    debug = False 
    auto_init_seed = False 

    seed_stack = SeedStack() 
    pred_centers = [] 
    trun_direction = False 
    close_visited_time = 0 
    valid_seed_count = 0 
    run_time_with_manual = 0 
    run_time_without_manual = 0 
    def get_saliency(self, gray, source):
        mean = np.mean(gray[source[0]-2 : source[0]+3, source[1]-2 : source[1]+3] )
        w  = np.abs(gray - mean) + 0.01
        w = (w-np.min(w)) / (np.max(w) - np.min(w))
        return w

    def gen_graph_from_grayvalue(self, gray, center):

        w = self.get_saliency(gray, center)

        G=nx.DiGraph()

        for row_idx in range(0,w.shape[0]):
            for col_idx in range(0,w.shape[1]):
                node_idx = row_idx * w.shape[1] + col_idx

                if row_idx > 0 and col_idx > 0:
                    node_topleft_idx = (row_idx-1) * w.shape[1] + col_idx-1
                    cost = w[row_idx-1,col_idx-1]
                    G.add_edge(node_idx,node_topleft_idx,weight=cost)

                if row_idx > 0:
                    node_top_idx = (row_idx-1)*w.shape[1] + col_idx
                    cost = w[row_idx-1,col_idx]
                    G.add_edge(node_idx,node_top_idx,weight=cost)

                if row_idx > 0 and col_idx < w.shape[1]-1:
                    node_topright_idx = (row_idx-1)*w.shape[1] + col_idx+1
                    cost = w[row_idx-1,col_idx+1]
                    G.add_edge(node_idx,node_topright_idx,weight=cost)

                if col_idx > 0:
                    node_left_idx = row_idx*w.shape[1] + col_idx-1
                    cost = w[row_idx,col_idx-1]
                    G.add_edge(node_idx,node_left_idx,weight=cost)

                if col_idx < w.shape[1]-1:
                    node_right_idx = row_idx*w.shape[1] + col_idx+1
                    cost = w[row_idx,col_idx+1]
                    G.add_edge(node_idx,node_right_idx,weight=cost)

                if row_idx < w.shape[0]-1 and col_idx > 0:
                    node_bottomleft_idx = (row_idx+1)*w.shape[1] + col_idx-1
                    cost = w[row_idx+1,col_idx-1]
                    G.add_edge(node_idx,node_bottomleft_idx,weight=cost)

                if row_idx < w.shape[0]-1:
                    node_bottom_idx = (row_idx+1)*w.shape[1] + col_idx
                    cost = w[row_idx+1,col_idx]
                    G.add_edge(node_idx,node_bottom_idx,weight=cost)

                if row_idx < w.shape[0]-1 and col_idx < w.shape[1]-1:
                    node_bottomright_idx = (row_idx+1)*w.shape[1] + col_idx+1
                    cost = w[row_idx+1,col_idx+1]
                    G.add_edge(node_idx,node_bottomright_idx,weight=cost)
        return G

    def direction_estimation(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

        gray = cv2.bilateralFilter(gray, 3, 3 * 2, 3 / 2)
        edge = cv2.Canny(gray,100,200,3)

        fft2 = np.fft.fft2(edge)
        shift2center = np.fft.fftshift(fft2)
        log_shift2center = np.log(1 + np.abs(shift2center))

        sumfft = np.zeros(180)
        row, col = gray.shape
        R = row if row < col else col
        R //= 2
        y0 = row // 2
        x0 = col // 2
        for theta in range(180):
            sumfft[theta] = 0
            for r in range(R):
                x = int(x0 + r * np.cos(theta * np.pi / 180.0))
                y = int(y0 + r * np.sin(theta * np.pi / 180.0))
                sumfft[theta] += log_shift2center[y, x]

        angle = np.argmax(sumfft)

        if self.visualize_step:
            x = int(x0 + R * np.cos(angle*np.pi / 180))
            y = int(y0 + R * np.sin(angle*np.pi / 180))
            print(np.max(log_shift2center))
            print(np.min(log_shift2center))
            log_shift2center = (log_shift2center-np.min(log_shift2center)) / (np.max(log_shift2center) - np.min(log_shift2center)) * 255
            log_shift2center = log_shift2center.astype(np.uint8)
            rgb = cv2.cvtColor(log_shift2center, cv2.COLOR_GRAY2RGB)
            cv2.line(rgb, (x0,y0), (x,y),(0,0,255), 1, 8, 0)
            plt.figure()
            plt.subplot(131)
            plt.imshow(gray, cmap='gray')
            plt.scatter(32, 32)
            plt.subplot(132)
            plt.imshow(edge, cmap='gray')
            plt.subplot(133)
            plt.imshow(rgb),plt.show()

        angle += 90
        return angle

        pass

    def get_next_points(self, patch, seed):
        angle = self.direction_estimation(patch.img)
        print('angle=', angle) if self.debug else 1
        y0 = seed[0] - patch.top
        x0 = seed[1] - patch.left
        p0, p1 = np.zeros(2),np.zeros(2)
        p0[0] = (y0 + self.step_size * np.sin(angle * np.pi / 180 ) + patch.top)
        p0[1] = (x0 + self.step_size * np.cos(angle * np.pi / 180) + patch.left)
        p1[0] = (y0 - self.step_size * np.sin(angle * np.pi / 180) + patch.top)
        p1[1] = (x0 - self.step_size * np.cos(angle * np.pi / 180) + patch.left)

        return (p0.astype(np.int), p1.astype(np.int))

    def get_min_distance(self, point):
        r, c = point
        min_dis = np.inf
        min_point = None
        for path in self.visited_path:
            for point in path:
                distance = math.sqrt(math.pow(point[0] - r, 2) + math.pow(point[1] - c, 2))
                if distance < min_dis:
                    min_dis = distance
                    min_point = point

        return min_dis, min_point

    def get_improve_morph_edge(self, gray):

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
        m_open = cv2.erode(gray, kernel)
        m_open = cv2.dilate(m_open, kernel)
        m_open = cv2.dilate(m_open, kernel)

        m_close = cv2.dilate(gray, kernel)
        m_close = cv2.erode(m_close, kernel)
        m_close = cv2.erode(m_close, kernel)

        m_edge = m_open - m_close

        if self.visualize_step:
            plt.figure()
            plt.subplot(131)
            plt.imshow(m_open,cmap='gray')
            plt.subplot(132)
            plt.imshow(m_close,cmap='gray')
            plt.subplot(133)
            plt.imshow(m_edge,cmap='gray')
            plt.show()

        return m_edge

    def get_adjust_seed(self, seed):
        patch_size = self.crop_size // 1
        box_top = seed[0] - patch_size // 2
        box_left = seed[1] - patch_size // 2
        box_width, box_height = patch_size, patch_size
        img_width, img_height, _ = self.np_image.shape
        [box_top, box_left] = self.adjust_box(box_top, box_left, box_width, box_height)

        patch = self.np_image[box_top:box_top + box_height, box_left:box_left + box_width, :]
        gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        gray = cv2.bilateralFilter(gray, 3, 3 * 2, 3 / 2)
        gray = cv2.bilateralFilter(gray, 3, 3 * 2, 3 / 2)
        m_edge = cv2.Canny(gray,100,200,3)
        m_edge = m_edge / 255.


        seed = np.array([seed[0]-box_top, seed[1]-box_left])


        r = 1
        point = np.zeros(2, dtype=np.int)
        while(True):
            min_t = 100 
            for i in range(-1,2):
                for j in range(-1,2):
                    seed_t = np.array([seed[0] + i, seed[1] + j])
                    sum_t = 0
                    for m in range(-r, r+1):
                        for n in range(-r, r+1):
                            ed = int(np.sqrt(m*m+n*n) + 0.5)
                            if ed > r:
                                continue
                            point[0] = seed_t[0] + m
                            point[1] = seed_t[1] + n

                            sum_t += m_edge[point[0], point[1]]

                    if min_t > sum_t:
                        min_t = sum_t
                        seed = seed_t

            print('{}->{}'.format(r, min_t)) if self.debug else 1

            if self.visualize_step:
                fig = plt.figure()
                ax = fig.add_subplot(121)
                ax.imshow(m_edge, cmap='gray')
                cir1 = Circle(xy=([seed[1]+2, seed[0]+2]), radius=r, alpha=0.7)
                ax.add_patch(cir1)
                cir2 = Circle(xy=([seed[1]+2, seed[0]+2]), radius=1, alpha=1)
                ax.add_patch(cir2)

                ax = fig.add_subplot(122)
                ax.imshow(patch)
                ax.scatter(seed[1]+2, seed[0]+2, marker='x', color='r')
                plt.show()

            if min_t < r and r < 7:
                r += 1
            else:
                seed = np.array([seed[0]+box_top+2, seed[1]+box_left+2]).astype(np.int)
                break




        return seed

    def adjust_box(self, y_tmp, x_tmp, height, width):
        img_width, img_height = self.image_size
        x_tmp = 0 if x_tmp < 0 else x_tmp
        x_tmp = img_width-1-width if x_tmp + width >= img_width else x_tmp
        y_tmp = 0 if y_tmp < 0 else y_tmp
        y_tmp = img_height-1-height if y_tmp + height >= img_height else y_tmp
        return y_tmp, x_tmp

    def build_model(self):
        p = {}
        p['g_size'] = 64  
        p['useAug'] = 0  
        p['numHG'] = 2  
        p['trainBatch'] = 64  
        p['Block'] = 'ConvBlock'  
        p['inputRes'] = (64, 64)  
        p['GTmasks'] = 0 
        p['outputRes'] = (64, 64)  
        p['useRandom'] = 1  

        numHGScales = 4  
        net = nt.Net_SHG(p['numHG'], numHGScales, p['Block'], 128, 1)
        net.load_state_dict(torch.load(self.checkname,
                                   map_location=lambda storage, loc: storage))
        net.eval()

        if torch.cuda.is_available():
            self.gpu_id = 0
        else:
            self.gpu_id = -1

        if self.gpu_id >= 0:
            torch.cuda.set_device(device=self.gpu_id)
            net.cuda()

        return net

    def gen_box_image_crop(self, seed, next_center):
        height, width, _ = self.np_image.shape
        radius = self.crop_size // 2
        min_r = seed[0] if seed[0] < next_center[0] else next_center[0]
        min_c = seed[1] if seed[1] < next_center[1] else next_center[1]
        max_r = seed[0] if seed[0] > next_center[0] else next_center[0]
        max_c = seed[1] if seed[1] > next_center[1] else next_center[1]

        ltr = 0 if min_r - radius < 0 else min_r - radius
        ltc = 0 if min_c - radius < 0 else min_c - radius
        rbr = height-1 if max_r + radius >= height else max_r + radius
        rbc = width-1 if max_c + radius >= width else max_c + radius

        patch = self.np_image[ltr : rbr+1, ltc : rbc+1, :]

        return (ltr, ltc), (rbr, rbc), patch

    def gen_visited_path(self, pathes):
        if self.visited_path is None:
            self.visited_path = []


        for path in pathes:
            self.visited_path.append(path)

    def save_results_fun(self, img_filename, path_mask, path):
        if not os.path.exists(self.save_topology_root):
            os.makedirs(self.save_topology_root)

        filename, _ = os.path.splitext(img_filename)

        img = Image.open(os.path.join(self.root_dir, img_filename))
        img = img.convert('RGBA')

        h, w = path_mask.shape
        img_mask = np.zeros((3, h, w), dtype=np.uint8)
        img_mask[:,:,:] = path_mask * 255
        img_mask = Image.fromarray(img_mask, mode='RGBA')
        img_mask.save(os.path.join(self.save_topology_root), filename + '_mask.png')

        img_overlay = Image.alpha_composite(img, img_mask)
        img_overlay.save(os.path.join(self.save_topology_root), filename + '_over.png')

        path.save(os.path.join(self.save_topology_root), filename + '_path.npy')

    def get_most_confident_outputs(self, patch):
        np_image_crop = patch.img.transpose((2, 0, 1)).astype(np.float32)
        np_image_crop = np.expand_dims(np_image_crop, axis=0)
        inputs = np_image_crop / 255 - 0.5
        _, _, GH, GW = inputs.shape

        inputs = torch.from_numpy(inputs)
        if self.gpu_id >= 0:
            inputs = inputs.cuda()

        with torch.no_grad():
            output = self.net.forward(inputs)

        pred = np.squeeze(np.transpose(output[len(output)-1].cpu().data.numpy()[0, :, :, :], (1, 2, 0)))




        mean, median, std = sigma_clipped_stats(pred, sigma=3.0)
        threshold = median + (10.0 * std)
        sources = find_peaks(pred, threshold, box_size=3)

        data = {}
        data['peaks'] = []
        indxs = np.argsort(sources['peak_value'])
        centers_idx = []
        for ii in range(0,len(indxs)):
            idx = indxs[len(indxs)-1-ii]
            data['peaks'].append({'x': int(sources['x_peak'][idx]), 'y': int(sources['y_peak'][idx]), 'value' : float(sources['peak_value'][idx])})
            index = sources['y_peak'][idx] * self.crop_size + sources['x_peak'][idx]
            centers_idx.append(index)

        points=[]
        confidences = []

        print('---------------------------------') if self.debug else 1
        for obj in data['peaks']:
            v = obj['value']
            print('confidence is ', v) if self.debug else 1
            if v < self.confidence_th:
                print('confidence is too low', v, ' and the confident threshold is ', self.confidence_th) if self.debug else 1
                break 

            point = [obj['y']+patch.top, obj['x']+patch.left]
            point = np.array(point, dtype=np.int)
            points.append(point)
            confidences.append(v)


        if self.visualize_step:
            plt.figure()
            plt.subplot(131)
            plt.imshow(patch.img)
            plt.subplot(132)
            plt.imshow(patch.img)
            c = ['red', 'yellow', 'green', 'blue', 'pink']
            i=0
            for point in points:
                plt.scatter(point[1]-patch.left,point[0]-patch.top, marker='o', color=c[i % 5])
                i+=1
                print(point[1]-patch.left,point[0]-patch.top)

            plt.subplot(133)
            plt.imshow(pred, cmap=plt.cm.hot_r)

            plt.show()

        return points, confidences

    def get_graydist(self, source, target):
        lt, rb, img_crop = self.gen_box_image_crop(source, target)
        gray = cv2.cvtColor(img_crop, cv2.COLOR_RGB2GRAY)
        G = self.gen_graph_from_grayvalue(gray, source)
        GH, GW = gray.shape
        target_idx = (target[0] - lt[0]) * GW + target[1]-lt[1]
        source_idx = (source[0] - lt[0]) * GW + source[1]-lt[1]
        length, path = nx.bidirectional_dijkstra(G,source_idx,target_idx)

        path_t = []
        for node in path:
            node_global_r = node // GW + lt[0]
            node_global_c = node % GW + lt[1]
            path_t.append([node_global_r, node_global_c])

        return length, path_t

        pass

    def crop_image(self, center, width, height):
        box_top = center[0] - height // 2
        box_left = center[1] - width // 2
        box_top, box_left = self.adjust_box(box_top, box_left, self.crop_size, self.crop_size)
        if len(self.np_image.shape) == 3:
            np_image_crop = self.np_image[box_top : box_top + self.crop_size, box_left : box_left + self.crop_size, :]
        elif len(self.np_image.shape) == 2:
            np_image_crop = self.np_image[box_top : box_top + self.crop_size, box_left : box_left + self.crop_size]


        return box_top, box_left, np_image_crop

    def auto_seed(self, fullyauto=False):
        from Progress import  Progress

        w, h = self.image_size

        patch_count = ((w-self.crop_size * 2 -1) // self.crop_size +1) * ((h-self.crop_size * 2 -1) // self.crop_size +1)

        if not fullyauto:
            progress = Progress(0,patch_count)

        self.seed_stack.clear()

        for r in range(self.crop_size, h-self.crop_size, self.crop_size):
            for c in range(self.crop_size, w-self.crop_size, self.crop_size):
                seed_t = [r + self.crop_size // 2, c + self.crop_size // 2]
                box_top, box_left, np_image_crop = self.crop_image(seed_t, self.crop_size, self.crop_size)
                patch = Patch(box_top, box_left, np_image_crop)
                [points,confidences] = self.get_most_confident_outputs(patch)
                if len(points) > 0:
                    if confidences[0] > self.auto_seed_confidence_th:
                        if self.auto_init_seed:
                            seed_t, pts = self.init_seed(points[0], False)
                            for pt in pts:
                                self.seed_stack.append((seed_t, pt, 1))
                        else:
                            self.seed_stack.append((points[0], None, 1))

                if not fullyauto:
                    progress.step()

        if not fullyauto:
            progress.destroy()

        self.seed_stack.reverse()

        return self.seed_stack

    def init_seed(self, seed, is_adjust):
        self.trun_direction = False
        h, w, _ = self.np_image.shape
        if is_adjust:
            seed = self.get_adjust_seed(seed)

        box_top, box_left, np_image_crop = self.crop_image(seed, self.crop_size, self.crop_size)
        patch = Patch(box_top, box_left, np_image_crop)

        points = self.get_next_points(patch, seed)

        return seed, points

    def get_line_path(self, seed, pred):
        y0, x0 = seed
        y1, x1 = pred
        path = []
        if x0 == x1:
            for y in range(y0, y1, 1 if y0<y1 else -1):
                path.append(np.round([y, x0],0).astype(np.int))
        else:
            k = (y0-y1) / (x0-x1)
            len = np.sqrt(np.power(y1-y0,2) + np.power(x1-x0,2))
            delta = (x1-x0) / len
            x = 0
            while True:
                y = k * x
                path.append(np.round([y+y0, x+x0],0).astype(np.int))
                x += delta
                if (delta >0 and x < (x1-x0)) or (delta < 0 and x > (x1-x0)):
                    continue
                else:
                    break

        path.append(pred)
        return path

    def get_vertical_extents(self, seed, next_center):
        node1 = next_center
        node0 = seed

        delty = node1[0] - node0[0]
        deltx = node1[1] - node0[1]
        if deltx == 0:
            newnode0 = np.array([node0[0], node0[1] + self.step_size]).astype(np.int)
            newnode1 = np.array([node0[0], node0[1] - self.step_size]).astype(np.int)
        elif delty == 0:
            newnode0 = np.array([node0[0] + self.step_size, node0[1]]).astype(np.int)
            newnode1 = np.array([node0[0] - self.step_size, node0[1]]).astype(np.int)
        else:
            k = delty / deltx
            newk = -1 / k
            theta = np.arctan(newk)
            newnode0 = np.array([node0[0] - self.step_size * np.sin(theta),
                                node0[1] - self.step_size * np.cos(theta)]).astype(np.int)
            newnode1 = np.array([node0[0] + self.step_size * np.sin(theta),
                                node0[1] + self.step_size * np.cos(theta)]).astype(np.int)

        return newnode0, newnode1

    def get_angle_from_three_coords(self, a, b, c):
        ab = self.get_len_from_two_points(a,b)
        bc = self.get_len_from_two_points(b,c)
        ac = self.get_len_from_two_points(a,c)
        if bc==0 or ab ==0: 
            thetaB = np.pi
        else:
            thetaB = np.arccos((ab * ab + bc * bc - ac * ac) / (2 * ab * bc))
        return thetaB

    def get_len_from_two_points(self, a, b):
        return np.sqrt(np.power(a[0]-b[0],2) + np.power(a[1]-b[1],2))

    def next_iter(self):
        next_center = None
        seed, next_center, ismanual = self.seed_stack.pop()

        while next_center is None:
            dist = self.get_min_distance(seed)
            if dist[0]<=self.close_visited_th:
                print('种子点', seed, '已经被跟踪过了') if self.debug else 1
            else:
                seed_t, points_t = self.init_seed(seed, False)
                for pt in points_t:
                    self.seed_stack.append((seed_t, pt, 1)) 

            if len(self.seed_stack)>0:
                seed, next_center, ismanual = self.seed_stack.pop()
            else:
                return None

        box_top, box_left, np_image_crop = self.crop_image(next_center, self.crop_size, self.crop_size)
        patch = Patch(box_top, box_left, np_image_crop)

        [points,confidences] = self.get_most_confident_outputs(patch)

        pred_center = None
        first_close_visited = False
        if len(points) == 0:
            return None
        else:
            prob_points = []
            next_center_rev = np.array([2*seed[0]-next_center[0], 2*seed[1]-next_center[1]])
            for i in range(0, len(points)):
                pt = points[i]
                confidence = confidences[i]
                theta_nextcenter_seed_pt = self.get_angle_from_three_coords(next_center_rev, seed, pt)
                prob_points.append([pt, theta_nextcenter_seed_pt, confidence])

            for i in range(len(prob_points) -1):
                for j in range(i+1,len(prob_points)):
                    if prob_points[i][1] < prob_points[j][1]:
                        tmp = prob_points[i]
                        prob_points[i] = prob_points[j]
                        prob_points[j] = tmp
            for i in range(0, len(prob_points)):
                pt = prob_points[i][0]
                distance = self.get_len_from_two_points(seed, pt)
                if distance>self.step_size / 10:
                    pred_center = pt
                    break

            if pred_center is not None: 
                for prob_point in prob_points:
                    if prob_point[1] * 180 / np.pi > 170: 
                        continue
                    if self.get_len_from_two_points(pred_center, prob_point[0]) <= self.step_size / 5:
                        continue
                    if self.get_min_distance(prob_point[0])[0] < self.close_visited_th:
                        continue
                    if prob_point[2] < np.max(confidences) * 0.5:
                        continue

                    ret = False
                    if self.auto_init_seed:
                        seed_t, points_t = self.init_seed(prob_point[0], False) 
                        for point_t in points_t:
                            seed_t = seed_t.astype(np.int)
                            point_t = point_t.astype(np.int)
                            ret = self.seed_stack.append((seed_t, point_t, 0))
                    else:
                        ret = self.seed_stack.append((prob_point[0], None, 0))

                    if ret :
                        print('auto find extend seed',prob_point[0]) if self.debug else 1

            if pred_center is None: 
                if self.trun_direction == False: 
                    print('the model want to retreat, try to extent from the perpendicular direction') if self.debug else 1
                    self.trun_direction = True
                    newnode0, newnode1 = self.get_vertical_extents(seed, next_center)
                    self.seed_stack.append((seed, newnode0, 0))
                    self.seed_stack.append((seed, newnode1, 0))
                return None
            else:
                min_dist, min_point = self.get_min_distance(pred_center)
                if min_dist <= self.close_visited_th: 
                    self.close_visited_time += 1
                    print(pred_center, 'meet the visited roads，the ', self.close_visited_time , ' times continuiously') if self.debug else 1
                    if self.close_visited_time>1: 
                        return None
                    else:
                        first_close_visited = True

        if pred_center is None:
            print('Unexpected situation, stop tracking') if self.debug else 1
            return None

        delta_r = pred_center[0] - seed[0]
        delta_c = pred_center[1] - seed[1]

        step_tmp = math.sqrt(math.pow(delta_r,2) + math.pow(delta_c, 2))
        delta_r = int(delta_r * (self.step_size / step_tmp))
        delta_c = int(delta_c * (self.step_size / step_tmp))
        next_center = np.array([pred_center[0] + delta_r, pred_center[1] + delta_c], dtype=np.int)

        if not first_close_visited:
            self.close_visited_time = 0

        self.trun_direction = False 

        if ismanual == 1:
            self.valid_seed_count += 1

        pathes =[]
        if self.use_line_conn:
            path_t = self.get_line_path(seed, pred_center)
            pathes.append(path_t)
        else:
            _, path_t = self.get_graydist(seed, pred_center)
            pathes.append(path_t)

        self.seed_stack.append((pred_center, next_center, 0)) 

        self.gen_visited_path(pathes)

        self.pred_centers.append(pred_center)


        return (pred_center, next_center, pathes)

    def __init__(self, imagename):

        self.net = self.build_model()
        _, self.img_filename = os.path.split(imagename)

        self.image = Image.open(imagename)

        self.image_size = self.image.size

        self.np_image = np.array(self.image)
        self.run_time_with_manual = time.time()
        self.run_time_without_manual = 0
