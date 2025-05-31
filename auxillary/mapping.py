#%%

import numpy as np
from matplotlib import pyplot as plt
from skimage.measure import LineModelND, ransac
from random import random
import cv2
from skimage import transform
import itertools as it
import time
import sklearn

import scipy as sp
import os
from tqdm import tqdm

from mpl_toolkits.mplot3d.art3d import Poly3DCollection, Line3DCollection

import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d
import numpy as np
from matplotlib.patches import Rectangle


#%% Functions from @Mateen Ulhaq and @karlo
def set_axes_equal(ax: plt.Axes):
    """Set 3D plot axes to equal scale.

    Make axes of 3D plot have equal scale so that spheres appear as
    spheres and cubes as cubes.  Required since `ax.axis('equal')`
    and `ax.set_aspect('equal')` don't work on 3D.
    """
    limits = np.array([
        ax.get_xlim3d(),
        ax.get_ylim3d(),
        ax.get_zlim3d(),
    ])
    origin = np.mean(limits, axis=1)
    radius = 0.5 * np.max(np.abs(limits[:, 1] - limits[:, 0]))
    _set_axes_radius(ax, origin, radius)

def _set_axes_radius(ax, origin, radius):
    x, y, z = origin
    ax.set_xlim3d([x - radius, x + radius])
    ax.set_ylim3d([y - radius, y + radius])
    ax.set_zlim3d([z - radius, z + radius])


def PiInv(points):
    return np.vstack((points, np.ones(points.shape[1])))

def Pi(points):
    return points[:-1]/points[-1]

def projectpoints(K,R,t,Q):
    rt = np.hstack((R,t))
    Q = PiInv(Pi(Q))
    return (K @ rt) @ Q
def box3d(n=16):
    points = []
    N = tuple(np.linspace(-1,1,n))
    for i,j in [(-1,-1), (-1,1), (1,1), (0,0)]:
        points.extend(set(it.permutations([(i, )*n, (j, )*n, N])))
    return np.hstack(points)/2



def unnormalize_bbox(bbox, im_shape=(1080,1920)):
    img_h = im_shape[0]
    img_w = im_shape[1]
    
    #print(bbox)
    xc, yc, wc, hc = bbox[1], bbox[2], bbox[3], bbox[4]
    
    w = wc*img_w
    h = hc*img_h
    
    x = xc*img_w -w/2
    y = yc*img_h -h/2
    
    return bbox[0],x,y,w,h

def CrossOp(p):
    """
    p: inhomogenous 3D point
    Anvende CrossOp operatoren for at lave en vektor om til en 
    3x3 matrix der kan fungere som krydsprodukt med en anden vektor
    """
    p = np.array(p).reshape(3,1)
    x = p[0][0]
    y = p[1][0]
    z = p[2][0]
    return np.array([[0,-z,y],[z,0,-x],[-y,x,0]])

def pest(Q,q):
    """
    Estimates the projection matrix from Q and q using
    Direct Linear Transformation method.
    Q (4xN): list of homogenous 3D points (real coordinates)
    q (3xN): list of homogenous 2D points (pixel coordinates)
    """
    
    B = []
    for i in range(len(Q.T)):
        Bi = np.kron( Q[:,i] , CrossOp( q[:,i]/q[-1,i] ) )
    
        if i == 0: B = Bi
        else: B = np.vstack((B,Bi))
    
    u,s,vh = np.linalg.svd(B)
    v = vh.T
    P = v[:,-1] # last has smallest eigenvalues

    P = P.reshape(4,3).T
    return P

def kronish(point):
    """
    Takes a point and constructs the cross-up of that point.
    """
    x = point[0]
    y = point[1]
    return np.array([[0,-1,y],[1,0,-x],[-y,x,0]])

def normalize2D(q):
    """
    Normalizes an array of N 2D inhomogeneous points.
    Maps them to homogeneous points.

    Parameters
    ----------
    q : np.array, shape(2,N)
        Array of N 2D points.

    Returns
    -------
    q_norm : np.array, shape(3,N)
             Array of N 2D normalized homogeneous points
    
    T : np.array, shape(3x3)
        The transformation matrix.

    """
    q = Pi(q)
    mu = np.mean(q,axis=1)
    sigma = np.std(q,axis=1)
    # print(mu)
    # print(sigma)
    
    Tinv = np.array([[sigma[0],0,mu[0]],[0,sigma[1],mu[1]],[0,0,1]])
    T = np.linalg.inv(Tinv)
    
    # print(T)
    
    return T@PiInv(q),T # q normalized homogenous, T

def hest(q1,q2,normalize=False):
    """
    Estimates a homography matrix from two sets of points.
    q1 lists of 2D homogenous points from source 1
    q2 lists of 2D homogenous points from source 2
    """
    if normalize:
        q1, T1 = normalize2D(q1)
        q2, T2 = normalize2D(q2)
        
    B = []
    for i in range(q2.shape[1]): # amount points
        Bi = np.kron( q2[:,i].T, kronish(q1[:,i]) )
        
        if i == 0: B = Bi
        else: B = np.vstack((B,Bi))

    u,s,vh = np.linalg.svd(B)
    v = vh.T
    Ht = v[:,-1]
    H = Ht.reshape(3,3).T
    
    if normalize: 
        H = (np.linalg.inv(T1) @ H @ T2)
        H /= np.linalg.norm(H,ord='fro')
        
    return H

#%% 

class HomographyMapping:
    def __init__(self, label, data, thres, im, ball_data, datanum, savepath):
        self.label = label          # for frontview, t for topview, 45 for 45 degree view Used for plotting only
        self.data = data            # dot data
        self.thres = thres          # threshold for ransac
        self.im = im                # image to be transformed
        self.ball_data = ball_data  # ball data
        self.datanum = datanum      # image index, used for saving files

        self.viewtype = 'UNKNOWN'
        self.h, self.w = self.im.shape[1],self.im.shape[0]

        bbox_w = max(data[:,0]) - min(data[:,0])
        bbox_h = max(data[:,1]) - min(data[:,1])
        
        bound = 0.5
        self.bbox = [min(data[:,0])-(bbox_w*bound),
                     max(data[:,0])+(bbox_w*bound),
                     min(data[:,1])-(bbox_h*bound),
                     max(data[:,1])+(bbox_h*bound)]

        self.savepath = savepath
        self.usedots = True # if False, only corners are used for mapping
        self.final_dist_error = None

        self.colors = ['r','g','b','y']
        self.colorsExt = ['r','g','b','y','m','c']

    def extract_lines(self):
        self.lines = []
        self.newdata = self.data.copy()
        
        self.standard_line = np.arange(0, self.h)
        
        
        
        self.inlier_list = []
        for k in range(4):
            # fit line using all data
            model = LineModelND()
            model.estimate(self.newdata)
            try: 
                model_robust, inliers = ransac(self.newdata, LineModelND, min_samples=2, residual_threshold=self.thres, max_trials=1000)
            except:
                print('this should not happen, right??')
                return None

            outliers = (inliers == False)
            # stack = np.hstack((self.newdata[inliers],(np.ones(len(self.newdata[inliers]))*k).reshape(-1,1) ))
            self.inlier_list.append(self.newdata[inliers])
            self.inliercount = sum(inliers)
            
            # generate coordinates of estimated models
            line_y_robust = model_robust.predict_y(self.standard_line)
            self.lines.append(line_y_robust)
            self.newdata = self.newdata[outliers]
        
        slopes = [l[1]-l[0] for l in self.lines]
        # print(slopes)
        
        linepoints = [] 
        for i in range(4):
            linepoints.append([(self.standard_line[0],self.lines[i][0]),(self.standard_line[-1],self.lines[i][-1])])
 
# INTERSECTIONS   
        def line_intersection(line1, line2):
            xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
            ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

            def det(a, b):
                return a[0] * b[1] - a[1] * b[0]

            div = det(xdiff, ydiff)
            if div != 0:
                d = (det(*line1), det(*line2))
                x = det(d, xdiff) / div
                y = det(d, ydiff) / div
                return x, y
            else:
                return 0,0
        
        def inside(point, bbox):
            return (bbox[0]<point[0]) & (point[0]<bbox[1]) & (bbox[2]<point[1]) & (point[1]<bbox[3])
        
        intersections = []
        for i in range(4):
            for j in range(i):
                p = line_intersection(linepoints[i],linepoints[j])
                if inside(p,self.bbox):
                    intersections.append(p)
        
        self.intersections = np.array(intersections)
        
        dx = abs(self.intersections[-1,0]-self.intersections[-2,0])
        dy = abs(self.intersections[-1,1]-self.intersections[-2,1])
        
        if dy < 100: self.viewtype = 'FRONTVIEW'
        elif dx < 100: self.viewtype = 'TOPVIEW'
        else: self.viewtype = '45VIEW'
        
        output = []
        inter_org = np.hstack((self.intersections,np.array([0,1,2,3]).reshape(1,-1).T))
        sorted_y = inter_org[inter_org[:,1].argsort()]
        
        largest_x_idx1 = np.argmax(sorted_y[:2][:,0])
        smallest_x_idx1 = np.argmin(sorted_y[:2][:,0])
        
        output.append(sorted_y[largest_x_idx1])
        output.append(sorted_y[smallest_x_idx1])
        
        largest_x_idx2 = np.argmax(sorted_y[2:][:,0])+2
        smallest_x_idx2 = np.argmin(sorted_y[2:][:,0])+2
        
        output.append(sorted_y[largest_x_idx2])
        output.append(sorted_y[smallest_x_idx2])
        output = np.array(output)
        
        if self.viewtype == 'TOPVIEW':
            output = np.vstack((output[1],output[3],output[0],output[2]))
        
        self.intersections_sorted = output[:,:2]
        order = list(map(int, output[:,2]))
        
        ### from here, intersections should be sorted such that
        #
        #      R            B
        #
        #      G            Y
        
        # print('intersection order:\t',order)
            
## dots
        # dot_list_raw = [self.inlier_list[int(o)] for o in order]
        
        dot_n = [np.hstack((self.inlier_list[i],np.ones(len(self.inlier_list[i])).reshape(-1,1)*i)) for i in range(4)]
        
        meanvals = np.array([np.mean(d,axis=0) for d in dot_n])
        mean_min = np.argmin(meanvals,axis=0)
        mean_max = np.argmax(meanvals,axis=0)
        
        if self.viewtype == 'TOPVIEW':
            first = dot_n[mean_min[1]] #min mean y value
            second = dot_n[mean_max[1]] #max mean y value
            fourth = dot_n[mean_min[0]] #min mean x value
            third = dot_n[mean_max[0]] #max mean x value
            
            first = first[first[:,0].argsort()]
            second = second[second[:,0].argsort()]
            third = third[third[:,1].argsort()][::-1]
            fourth = fourth[fourth[:,1].argsort()][::-1]
            
        elif self.viewtype == 'FRONTVIEW':
            fourth = dot_n[mean_min[1]] #min mean y value
            third = dot_n[mean_max[1]] #max mean y value
            second = dot_n[mean_min[0]] #min mean x value
            first = dot_n[mean_max[0]] #max mean x value
            
            first = first[first[:,1].argsort()]
            second = second[second[:,1].argsort()]
            third = third[third[:,1].argsort()]
            fourth = fourth[fourth[:,1].argsort()]
            
            
        elif self.viewtype == '45VIEW':
            dot_n = [np.hstack((dot_n[i],np.ones(len(dot_n[i])).reshape(-1,1)*i)) for i in range(4)]

            meanvals = np.array([np.mean(d,axis=0) for d in dot_n])
            mean_min = np.argmin(meanvals,axis=0)
            mean_max = np.argmax(meanvals,axis=0)

            sorted_x = meanvals[meanvals[:,0].argsort()]

            minorx = sorted_x[:2]
            majorx = sorted_x[2:]
            
            if slopes[0] > 0:
                second = dot_n[int(minorx[np.argmax(minorx[:,1])][2])]
                fourth = dot_n[int(minorx[np.argmin(minorx[:,1])][2])]
                third = dot_n[int(majorx[np.argmax(majorx[:,1])][2])]
                first = dot_n[int(majorx[np.argmin(majorx[:,1])][2])]
                
                first = first[first[:,0].argsort()]
                second = second[second[:,0].argsort()]
                third = third[third[:,1].argsort()][::-1]
                fourth = fourth[fourth[:,1].argsort()][::-1]
                
            else:
                third = dot_n[int(minorx[np.argmax(minorx[:,1])][2])]
                second = dot_n[int(minorx[np.argmin(minorx[:,1])][2])]
                first = dot_n[int(majorx[np.argmax(majorx[:,1])][2])]
                fourth = dot_n[int(majorx[np.argmin(majorx[:,1])][2])]
            
                first = first[first[:,0].argsort()][::-1]
                second = second[second[:,0].argsort()][::-1]
                third = third[third[:,1].argsort()]
                fourth = fourth[fourth[:,1].argsort()]
        
        dot_list_n = [first,second,third,fourth]
        
        if [len(dot_list_n[i]) for i in range(4)] != [6,6,3,3]:
            print('Error in dot count, using only corners for mapping')
            self.usedots = False

        self.lineorder = [int(d[:,2][0]) for d in dot_list_n]
        # print('line order:\t\t',self.lineorder)
        
        self.dot_list = [d[:,:2] for d in dot_list_n]
        
        ### from here, intersections and dots should be sorted such that
        #
        #      R     r      B
        #      y            b
        #      G     g      Y
        
        

        if self.datanum == 11 and self.label == 'f': # dirty fix
                print("TODO: something is wrong in the sorting algorithm, fix this later")
                print(self.dot_list[3])
                movedot0 = self.dot_list[3][0].copy()
                movedot1 = self.dot_list[3][1].copy()
                self.dot_list[3][0] = movedot1
                self.dot_list[3][1] = movedot0
                print(self.dot_list[3])
        
    # final found points:
        if self.usedots: found_points = self.dot_list.copy()
        else: found_points = []
        found_points.append(self.intersections_sorted)
        
        self.found_points = np.vstack(found_points)
        
        ratio_1 = np.linalg.norm(self.intersections_sorted[0]-self.intersections_sorted[1],2)
        ratio_2 = np.linalg.norm(self.intersections_sorted[2]-self.intersections_sorted[3],2)
        
        ratio_3 = np.linalg.norm(self.intersections_sorted[0]-self.intersections_sorted[2],2)
        ratio_4 = np.linalg.norm(self.intersections_sorted[1]-self.intersections_sorted[3],2)
        
        angleratio = ((ratio_1+ratio_2)/(ratio_3+ratio_4)) * 5
        
        # print('ratios: {:.2f} {:.2f} {:.4f}'.format(ratio_1+ratio_2,ratio_3+ratio_4,angleratio))
    #### New template
        Y = 790
        X = 1472

        Cy,Cx = (Y//2,X//2)

        #dots
        space = 163.5
        h1 = 372
        h2 = 700

        dots_template = []
        if self.usedots:
            for i in range(7):
                if i != 3: dots_template.append([Cx+(i-3)*space,Cy+h1])
            for i in range(7):
                if i != 3: dots_template.append([Cx+(i-3)*space,Cy-h1])
            for i in range(3):
                dots_template.append([Cx+h2,Cy+(i-1)*space])
            for i in range(3):
                dots_template.append([Cx-h2,Cy+(i-1)*space])
        
        cornerpoints_template = [[Cx-h2,Cy+h1],[Cx-h2,Cy-h1],[Cx+h2,Cy+h1],[Cx+h2,Cy-h1]]
        
        cornerpoints_template = np.array(cornerpoints_template)
        dots_template = np.array(dots_template)
        
        if self.usedots: self.template_points = np.vstack((dots_template,cornerpoints_template))
        else: self.template_points = cornerpoints_template
        
        self.template_points = np.array([[i[0]+500,i[1]+500] for i in self.template_points])

    # HOMOGRPHY

        self.H = hest(PiInv(self.template_points.T), PiInv(self.found_points.T))
        
        # Apply the perspective transformation to the image
        self.warpedim = cv2.warpPerspective(self.im, self.H, (self.h*2,self.w*2))
        
        tform = transform.ProjectiveTransform(self.H)
        
        if len(self.ball_data[0]) > 2: 
            if self.label != "t": # used to do positioning of ballcenter of ball angleratio
                self.ball_centers = np.array([[b[0]+(b[2]/2),b[1]+(b[3])/angleratio] for b in self.ball_data])
            else:
                self.ball_centers = np.array([[b[0]+(b[2]/2),b[1]+(b[3]/2)] for b in self.ball_data])
        else: 
            self.ball_centers = np.array([[b[0],b[1]] for b in self.ball_data])
        
        self.ball_H = tform(self.ball_centers)
        
        self.H_dots = np.array([tform(dots) for dots in self.found_points]).reshape(-1,2)
        

    def plot_lines(self, save=True):
        
        fig, ax = plt.subplots(1,2,figsize=(15,8))
        # ax[0].set_title('Original Image before sorting points')
        # ax[1].set_title('Original Image after sorting points')
        
    #setup plots
        ax[0].imshow(self.im)
        ax[0].set_xlim([0, self.h])
        ax[0].set_ylim([self.w, 1])
        ax[0].axis('off')
        
        ax[1].imshow(self.im)
        ax[1].set_xlim([0, self.h])
        ax[1].set_ylim([self.w, 1])
        ax[1].axis('off')
        
    # plot ball points
        # ax[0].scatter(self.ball_centers[:,0], self.ball_centers[:,1], s=20, color='m',label=f'Balls: ({len(ball_data)})')
        # ax[1].scatter(self.ball_centers[:,0], self.ball_centers[:,1], s=20, color='m',label=f'Balls: ({len(ball_data)})')
        
        # grid = np.ones_like(out[:,:,0])
        # for i in dotsH[0][:,0]: grid[:,int(i)] = 0
        # for i in dotsH[3][:,1]: grid[int(i),:] = 0
        # grid[:,int(h/2)] = 0
        
        # ax[1].imshow(grid, cmap='binary_r', vmin=0, vmax=1)
        
    # plot unsorted
        num = 0
        for i in range(len(self.inlier_list)):
            # k missing here?!?!
            ax[0].plot(self.standard_line, self.lines[i], color=self.colors[i],linewidth=7,label=f'iter {i+1}: ({self.inliercount})',zorder=1)
            ax[0].text(self.intersections[i,0], self.intersections[i,1], ' '+chr(65+i)+' ', bbox = {'facecolor': 'oldlace', 'alpha': 1, 'boxstyle': "square,pad=0.8", 'ec': 'k'},zorder=2)
            for p in range(len(self.inlier_list[i])):
                # ax[0].scatter(self.inlier_list[k][:,0], self.inlier_list[k][:,1], s=50, color=self.colors[k],zorder=1)
                # ax[1].text(self.inlier_list[i][p,0], self.dot_list[i][p,1], f'{p}', bbox = {'facecolor': 'oldlace', 'alpha': 1, 'boxstyle': "circle,pad=0.3", 'ec': self.colors[i]})
                # ax[0].plot(self.standard_line, self.lines[k], color=self.colors[k],linewidth=3,label=f'iter {k+1}: ({self.inliercount})',zorder=1)
                ax[0].text(self.inlier_list[i][p,0], self.inlier_list[i][p,1], f'{p+num}', bbox = {'facecolor': 'oldlace', 'alpha': 1, 'boxstyle': "circle,pad=0.3", 'ec': self.colors[i]},zorder=3)
            num = len(self.inlier_list[i])+num+1
            
        if len(self.newdata)>0 : ax[0].scatter(self.newdata[:,0], self.newdata[:,1], s=30, color='k', label=f'outliers: ({self.newdata.shape[0]})',zorder=2)

    # plot sorted dotpoints
        num = 0
        for i in range(4):
            # k missing here?!?!
            ax[1].plot(self.standard_line, self.lines[self.lineorder[i]], color=self.colors[i],linewidth=3,label=f'iter {i+1}: ({self.inliercount})',zorder=1)
            ax[1].text(self.intersections_sorted[i,0], self.intersections_sorted[i,1], ' '+chr(65+i)+' ', bbox = {'facecolor': 'oldlace', 'alpha': 1, 'boxstyle': "square,pad=0.8", 'ec': 'k'},zorder=2)
            for p in range(len(self.dot_list[i])):
                # ax[1].scatter(self.dot_list[i][p,0], self.dot_list[i][p,1], s=60, c=colorsExt[i],zorder=2)
                # ax[1].scatter(self.dot_list[i][p,0], self.dot_list[i][p,1], s=30, c=colorsExt[p],zorder=3)
                
                ax[1].text(self.dot_list[i][p,0], self.dot_list[i][p,1], f'{p+num}', bbox = {'facecolor': 'oldlace', 'alpha': 1, 'boxstyle': "circle,pad=0.3", 'ec': self.colors[i]},zorder=3)
            num = p+num+1
            
        # ax[1].scatter(self.intersections_sorted[:,0], self.intersections_sorted[:,1], s=60, color=colors,zorder=3)
        # ax[0].legend(loc='center', bbox_to_anchor=(0.5, -0.07),ncol=4)
        
        ax[0].set_xlabel(f'Guess: {self.viewtype}')
        fig.tight_layout()

        if save: 
            fig.savefig(self.savepath+f'lines_{self.datanum}{self.label}.png', bbox_inches='tight',dpi=200)
        # plt.show()
        # plt.close()
    
    def plot_template(self, save=True):
        fig2, ax2 = plt.subplots(2,1,figsize=(10,10))
        
        ax2[0].set_title('Order of template points')
        ax2[1].set_title('Order of segmented points')
        
        ax2[0].scatter(self.template_points[:,0],self.template_points[:,1])
        ax2[1].scatter(self.found_points[:,0],self.found_points[:,1])
        
        ax2[0].set_ylim(min(self.template_points[:,1])-100,max(self.template_points[:,1])+100)
        ax2[1].set_ylim(max(self.found_points[:,1])+100,min(self.found_points[:,1])-100)
        ax2[0].set_aspect('equal', adjustable='box')
        ax2[1].set_aspect('equal', adjustable='box')
        
        for i in range(len(self.found_points)):
            ax2[0].annotate(' '+str(i), (self.template_points[i,0], self.template_points[i,1]))
            ax2[1].annotate(' '+str(i), (self.found_points[i,0], self.found_points[i,1]))
        
        if save: 
            plt.savefig(self.savepath+f'template_{self.datanum}{self.label}.png', bbox_inches='tight',dpi=200)
        # plt.show()
        # plt.close()
            
    def plot_compare2topview(self, compareballs, warpedim=None, classes=[], save=True):
        dist_error = 0

        if warpedim is None:
            print("No topview image provided")

        else:
            fig3, ax3 = plt.subplots(2,1,figsize=(10,10))
            # ax3 = [plt.subplots(1,1,figsize=(10,10))[1],plt.subplots(1,1,figsize=(6,6))[1]]
            ax3[0].set_title('Reference image')
            ax3[1].set_title('Warped positions using points')
            
            ax3[0].imshow(self.im)
            # ax3[1].imshow(out)
            
            ballsize = 140 # 4000
            dotsize= 35
            refsize = 150
            
            x1=min(self.H_dots[:,0])-20
            x2=max(self.H_dots[:,0])+20
            y1=min(self.H_dots[:,1])-20
            y2=max(self.H_dots[:,1])+20    
                
            if warpedim is not None: 
                # from skimage.color import rgb2gray
                # warpedim = rgb2gray(warpedim)

                ax3[1].imshow(warpedim) # ,cmap=plt.cm.gray
                ax3[1].add_patch(Rectangle((x1,y1),x2-x1,y2-y1,color='w',alpha=0.5))
                
            #     from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
            #     from mpl_toolkits.axes_grid1.inset_locator import mark_inset
                
            #     axins = zoomed_inset_axes(ax3[1], 6, loc=1) # zoom = 6
                
            #     extent = [-3, 4, -4, 3]
            #     axins.imshow(warpedim, extent=extent, interpolation="nearest",
            #                  origin="lower")
                
            #     # sub region of the original image
            #     x1, x2, y1, y2 = -1.5, -0.9, -2.5, -1.9
            #     axins.set_xlim(x1, x2)
            #     axins.set_ylim(y1, y2)
                
            #     plt.xticks(visible=False)
            #     plt.yticks(visible=False)
                
            #     # draw a bbox of the region of the inset axes in the parent axes and
            #     # connecting lines between the bbox and the inset axes area
            #     mark_inset(ax3[1], axins, loc1=2, loc2=4, fc="none", ec="0.5")
            
            ax3[0].scatter(self.inlier_list[0][:,0], self.inlier_list[0][:,1], s=60, facecolors='m', edgecolors='w',zorder=1,label='projection points')
            ax3[0].scatter(self.intersections_sorted[0,0], self.intersections_sorted[0,1], s=refsize, color='r',marker="+",label='reference mark')
            
            for k in range(1,4): 
                ax3[0].scatter(self.inlier_list[k][:,0], self.inlier_list[k][:,1], s=60, facecolors='m', edgecolors='w',zorder=1)
                ax3[0].scatter(self.intersections_sorted[k,0], self.intersections_sorted[k,1], s=60, facecolors='m', edgecolors='w',)
            
            # if len(self.newdata)>0 : ax[0].scatter(self.newdata[:,0], self.newdata[:,1], s=dotsize, color='grey',zorder=2)
            
            ax3[1].scatter(self.H_dots[-4,0], self.H_dots[-4,1],s=refsize, color='r',marker="+",zorder=3,label='reference mark')
            
            # print(classes)
            if classes == []:
                ax3[0].scatter(self.ball_centers[:,0], self.ball_centers[:,1], s=30, facecolors='y', edgecolors='k',label='found balls')
                ax3[1].scatter(self.ball_H[:,0], self.ball_H[:,1], s=ballsize, linewidth=4,facecolors='none', edgecolors='m',label='projected balls')
            
            else:
                classcolors = [(0,0,1),(0,1,0),(1,0,0),(0,1,1)]
                
                for ball in range(len(self.ball_centers)): # facecolors='none', edgecolors
                    # ax3[0].scatter(self.ball_centers[ball,0], self.ball_centers[ball,1], s=ballsize, facecolors='none', edgecolors=classcolors[int(classes[ball])])
                    ax3[1].scatter(self.ball_H[ball,0], self.ball_H[ball,1], s=ballsize, facecolors='none',edgecolors=classcolors[int(classes[ball])])
            
            
            ax3[1].scatter(self.H_dots[:-4,0], self.H_dots[:-4,1], s=dotsize, color='grey',zorder=1,label='projection points')
            ax3[1].scatter(self.H_dots[-3:,0], self.H_dots[-3:,1], s=dotsize, color='grey',zorder=1)
            
            if len(compareballs)>0 and self.label!="t":

                # b_new = self.ball_H[np.lexsort(np.transpose(self.ball_H)[::-1])]
                # b_org = compareballs[np.lexsort(np.transpose(compareballs)[::-1])]
                
                b_new = self.ball_H[np.lexsort(np.transpose(self.ball_H)[::-1])]
                b_org = compareballs[np.lexsort(np.transpose(compareballs)[::-1])]

                #dirty fix no 2:
                #balls not ordered correctly
                wrong = None
                if self.datanum == 1: wrong = [9,10]
                elif self.datanum == 23: wrong = [5,6]
                elif self.datanum == 25: wrong = [12,13]

                if wrong is not None:
                    moveball0 = b_new[wrong[0]].copy()
                    moveball1 = b_new[wrong[1]].copy()
                    b_new[wrong[0]] = moveball1
                    b_new[wrong[1]] = moveball0
                
                dist_error = np.sqrt(np.sum((b_new-b_org)**2,axis=1))
                
                # print("dist_error:",dist_error)
                # print("dist_error mean:",np.mean(dist_error))
                # print("lenght:",(np.max(self.H_dots,axis=0)[0]-np.min(self.H_dots,axis=0)[0]))
                m_l = np.mean(dist_error)/(np.max(self.H_dots,axis=0)[0]-np.min(self.H_dots,axis=0)[0])
                # print("dist_error mean / lenght:",m_l)
                self.final_dist_error = round(254*m_l,3)
                # print("=",self.final_dist_error,"cm")

                # dist in pixels, not checked might be wrong
                # numerator = (np.max(self.H_dots,axis=0)[0]-np.min(self.H_dots,axis=0)[0])
                # print([254* i/numerator for i in dist_error])
                
                ax3[1].plot([b_new[0,0],b_org[0,0]], [b_new[0,1],b_org[0,1]], 'b',linestyle="-", label='shift')
                for i in range(1,len(self.ball_H)):
                    ax3[1].plot([b_new[i,0],b_org[i,0]], [b_new[i,1],b_org[i,1]], 'b',linestyle="-")
                    # ax3[1].text(b_new[i,0], b_new[i,1], f'{i}', bbox = {'facecolor': 'oldlace', 'alpha': 1, 'boxstyle': "circle,pad=0.3", 'ec': 'b'})
                    # ax3[1].text(b_org[i,0]+100, b_org[i,1], f'{i}', bbox = {'facecolor': 'oldlace', 'alpha': 1, 'boxstyle': "circle,pad=0.3", 'ec': 'r'})
                    
            ax3[0].set_xlim(0,self.h)
            ax3[0].set_ylim(self.w,0)
            
            # ax3[1].set_xlim(0,im.shape[1]*1.5)
            # ax3[1].set_ylim(0,im.shape[0]*1.5)
            
            ax3[1].set_xlim(x1,x2) # see full table
            ax3[1].set_ylim(y1,y2)
            
            ax3[1].set_aspect('equal', adjustable='box')
            ax3[1].set_aspect('equal', adjustable='box')
            
            # ax3[0].axis('off')
            # ax3[1].axis('off')
            
            # ax3[0].legend(loc='center', bbox_to_anchor=(0.5, -0.07),ncol=4)
            # ax3[1].legend(loc='center', bbox_to_anchor=(0.5, -0.07),ncol=4)
            
            # fig3.tight_layout()
            
            if save: 
                if self.final_dist_error > 1:
                    self.savepath+='errors/'
                plt.savefig(self.savepath+f'shift_{self.label}_{self.final_dist_error}.png', bbox_inches='tight',dpi=200)
            
            plt.show()
            plt.close()

    def extract_things_we_might_need(self):
        return self.H, self.template_points, self.found_points, self.ball_H, self.final_dist_error, self.viewtype, self.warpedim

