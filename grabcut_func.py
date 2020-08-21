# -*- coding: utf-8 -*-
'''
GrabCut function
'''

import sys
import numpy as np
import cv2 as cv
import igraph as ig

from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

BLUE = [255, 0, 0]        # the color of drawing rectangle
RED = [0, 0, 255]         # PR BG
GREEN = [0, 255, 0]       # PR FG
BLACK = [0, 0, 0]         # sure BG
WHITE = [255, 255, 255]   # sure FG

DRAW_NULL = {'color': BLUE, 'val': -1}
DRAW_BG = {'color': BLACK, 'val': 0}
DRAW_FG = {'color': WHITE, 'val': 1}
DRAW_PR_FG = {'color': GREEN, 'val': 3}
DRAW_PR_BG = {'color': RED, 'val': 2}

GMM_COMPONENTS_DEFAULT = 5 # suggested value of GMM components kn
GAMMA_DEFAULT = 50

# here are two classes, grabcut and GMM. GMM is used for grabcut.

class grabcut:
	def __init__(self, img_input, mask_input, rect_input=None, gmm_components_kn_input=GMM_COMPONENTS_DEFAULT):
		
		# img_input: the given img
		self.img=np.asarray(img_input, dtype=np.float64)
		
		# mask_input: a matrix with the same size as img, each value of the mask corresponds a pixel in img indicating where it belongs to foreground or background
		self.mask=mask_input
		
		self.height, self.width, _=self.img.shape
		
		assert np.unique(mask_input)==DRAW_BG['val']
		if rect_input is not None:
			self.mask[rect_input[1]:(rect_input[1]+rect_input[3]),rect_input[0]:(rect_input[0]+rect_input[2])]=DRAW_PR_FG['val']

		self.get_fore_back_ground_pixel_location()

		# constant parameter setting
		self.gmm_components_kn=gmm_components_kn_input
		self.gamma=GAMMA_DEFAULT
		self.penalty_term_for_construct_graph=500

		self.graph_foreground_node=self.height*self.width
		self.graph_background_node=self.graph_foreground_node + 1

		# init GMM
		self.init_GMM()
        
        # calculate smoothness term
		self.calculate_smoothness_term()
		
        #pr_fg_num_last=np.sum(np.array(self.mask[self.mask==DRAW_PR_FG['val']], dtype=np.int))
		#pr_bg_num_last=np.sum(np.array(self.mask[self.mask==DRAW_PR_BG['val']], dtype=np.int))

		for i in range(1):

			# train GMM
			self.train_GMM()

			# estimate segmentation
			self.construct_graph_of_grabcut_and_estimate_segmentation()

			self.get_fore_back_ground_pixel_location()

	def draw_mask(self):
		for i in range(1):
			# calculate smoothness term
			#self.calculate_smoothness_term()

			# train GMM
			self.train_GMM()
				
			# estimate segmentation
			self.construct_graph_of_grabcut_and_estimate_segmentation()

			self.get_fore_back_ground_pixel_location()

	def get_fore_back_ground_pixel_location(self):
		self.foreground_pix_loc=np.where(np.logical_or(self.mask == DRAW_FG['val'], self.mask == DRAW_PR_FG['val']))
		self.background_pix_loc=np.where(np.logical_or(self.mask == DRAW_BG['val'], self.mask == DRAW_PR_BG['val']))

	def init_GMM(self):
		self.foreground_gmm = GMM(self.gmm_components_kn)
		self.background_gmm = GMM(self.gmm_components_kn)

	def train_GMM(self):
		self.foreground_gmm.data_input_and_fit_gmm(self.img[self.foreground_pix_loc])
		self.background_gmm.data_input_and_fit_gmm(self.img[self.background_pix_loc])
		print("Train GMM")
		print("foreground_pix: "+str(len(self.foreground_pix_loc[0])))
		print("background_pix: "+str(len(self.background_pix_loc[0])))

	def construct_graph_of_grabcut_and_estimate_segmentation(self):
		sure_foreground_pix_loc_reshape=np.where(self.mask.reshape(-1)==DRAW_FG['val'])
		sure_background_pix_loc_reshape=np.where(self.mask.reshape(-1)==DRAW_BG['val'])
		predict_pix_loc_reshape=np.where(np.logical_or(self.mask.reshape(-1)==DRAW_PR_BG['val'],self.mask.reshape(-1)==DRAW_PR_FG['val']))

		print('Before cut:')
		print('foreground pixel count: '+str(len(sure_foreground_pix_loc_reshape[0])))
		print('background pixel count: '+str(len(sure_background_pix_loc_reshape[0])))
		print('uncertain pixel count: '+str(len(predict_pix_loc_reshape[0])))
		print('\n')

		edge=[]
		self.grabcut_graph_edge_capacity=[]

		# link "foreground" and "background" nodes to other nodes
		edge.extend(list(zip([self.graph_foreground_node]*predict_pix_loc_reshape[0].size, predict_pix_loc_reshape[0])))
		D=(self.background_gmm.predict(self.img.reshape(-1, 3)[predict_pix_loc_reshape])).tolist()
		self.grabcut_graph_edge_capacity.extend(D)
		assert len(edge) == len(self.grabcut_graph_edge_capacity)

		edge.extend(list(zip([self.graph_background_node]*predict_pix_loc_reshape[0].size, predict_pix_loc_reshape[0])))
		D=(self.foreground_gmm.predict(self.img.reshape(-1, 3)[predict_pix_loc_reshape])).tolist()
		self.grabcut_graph_edge_capacity.extend(D)
		assert len(edge) == len(self.grabcut_graph_edge_capacity)

		edge.extend(list(zip([self.graph_foreground_node]*sure_foreground_pix_loc_reshape[0].size, sure_foreground_pix_loc_reshape[0])))
		D=[self.penalty_term_for_construct_graph]*sure_foreground_pix_loc_reshape[0].size
		self.grabcut_graph_edge_capacity.extend(D)
		assert len(edge) == len(self.grabcut_graph_edge_capacity)

		edge.extend(list(zip([self.graph_foreground_node]*sure_background_pix_loc_reshape[0].size, sure_background_pix_loc_reshape[0])))
		D=[0]*sure_background_pix_loc_reshape[0].size
		self.grabcut_graph_edge_capacity.extend(D)
		assert len(edge) == len(self.grabcut_graph_edge_capacity)

		edge.extend(list(zip([self.graph_background_node]*sure_foreground_pix_loc_reshape[0].size, sure_foreground_pix_loc_reshape[0])))
		D=[0]*sure_foreground_pix_loc_reshape[0].size
		self.grabcut_graph_edge_capacity.extend(D)
		assert len(edge) == len(self.grabcut_graph_edge_capacity)

		edge.extend(list(zip([self.graph_background_node]*sure_background_pix_loc_reshape[0].size, sure_background_pix_loc_reshape[0])))
		D=[self.penalty_term_for_construct_graph]*sure_background_pix_loc_reshape[0].size
		self.grabcut_graph_edge_capacity.extend(D)
		assert len(edge) == len(self.grabcut_graph_edge_capacity)

		# link other nodes to each other
		img_indexes=np.arange(self.height*self.width, dtype=np.uint32).reshape(self.height, self.width)

		img_direction_1=img_indexes[:, 1:].reshape(-1)
		img_direction_2=img_indexes[:, :-1].reshape(-1)
		edge.extend(list(zip(img_direction_1, img_direction_2)))
		D=self.right_left_V.reshape(-1).tolist()
		self.grabcut_graph_edge_capacity.extend(D)
		assert len(edge) == len(self.grabcut_graph_edge_capacity)

		img_direction_1=img_indexes[1:, 1:].reshape(-1)
		img_direction_2=img_indexes[:-1, :-1].reshape(-1)
		edge.extend(list(zip(img_direction_1, img_direction_2)))
		D=self.lowerright_upleft_V.reshape(-1).tolist()
		self.grabcut_graph_edge_capacity.extend(D)
		assert len(edge) == len(self.grabcut_graph_edge_capacity)

		img_direction_1=img_indexes[1:, :].reshape(-1)
		img_direction_2=img_indexes[:-1, :].reshape(-1)
		edge.extend(list(zip(img_direction_1, img_direction_2)))
		D=self.up_down_V.reshape(-1).tolist()
		self.grabcut_graph_edge_capacity.extend(D)
		assert len(edge) == len(self.grabcut_graph_edge_capacity)

		img_direction_1=img_indexes[1:, :-1].reshape(-1)
		img_direction_2=img_indexes[:-1, 1:].reshape(-1)
		edge.extend(list(zip(img_direction_1, img_direction_2)))
		D=self.upright_lowerleft_V.reshape(-1).tolist()
		self.grabcut_graph_edge_capacity.extend(D)
		assert len(edge) == len(self.grabcut_graph_edge_capacity)

		# constructing graph
		self.grabcut_graph=ig.Graph(self.height*self.width+2)
		self.grabcut_graph.add_edges(edge)

		# min cut
		mincut = self.grabcut_graph.st_mincut(self.graph_foreground_node, self.graph_background_node, self.grabcut_graph_edge_capacity)
		print("After cut:")
		print('foreground pixel count: %d\nbackground pixel count: %d' % (len(mincut.partition[0]), len(mincut.partition[1])))
		print('\n')
		predict_pix_loc=np.where(np.logical_or(self.mask == DRAW_PR_BG['val'], self.mask == DRAW_PR_FG['val']))
		self.mask[predict_pix_loc] = np.where(np.isin(img_indexes[predict_pix_loc], mincut.partition[0]), DRAW_PR_FG['val'], DRAW_PR_BG['val'])

	def calculate_smoothness_term(self):
		right_left_diff = self.img[:, 1:] - self.img[:, :-1]
		lowerright_upleft_diff = self.img[1:, 1:] - self.img[:-1, :-1]
		up_down_diff = self.img[1:, :] - self.img[:-1, :]
		upright_lowerleft_diff = self.img[1:, :-1] - self.img[:-1, 1:]
		
		# calculate beta
		self.beta_in_smoothness_term = 1 / (2 * (np.sum(np.square(right_left_diff))+np.sum(np.square(lowerright_upleft_diff))+\
			np.sum(np.square(up_down_diff))+np.sum(np.square(upright_lowerleft_diff))) / (4 * self.height * self.width - 3 * self.height - 3 * self.width + 2))

		# calculate smoothness term V defined in formula (11)
		self.right_left_V = self.gamma * np.exp(-self.beta_in_smoothness_term * np.sum(np.square(right_left_diff), axis=2))
		self.lowerright_upleft_V = self.gamma / np.sqrt(2) * np.exp(-self.beta_in_smoothness_term * np.sum(np.square(lowerright_upleft_diff), axis=2))
		self.up_down_V = self.gamma * np.exp(-self.beta_in_smoothness_term * np.sum(np.square(up_down_diff), axis=2))
		self.upright_lowerleft_V = self.gamma / np.sqrt(2) * np.exp(-self.beta_in_smoothness_term * np.sum(np.square(upright_lowerleft_diff), axis=2))


class GMM:
	def __init__(self, train_data, gmm_components_kn_input=GMM_COMPONENTS_DEFAULT):
		self.gmm_components_kn=gmm_components_kn_input
		self.estimator=GaussianMixture(n_components=self.gmm_components_kn,covariance_type='full', n_init=1, max_iter=100)
    
	def data_input_and_fit_gmm(self, train_data):
		self.estimator.fit(train_data)
		self.get_coefs_means_covariances()

	def get_coefs_means_covariances(self):
		self.coefs=self.estimator.weights_
		self.means=self.estimator.means_
		self.covariances=self.estimator.covariances_

	def predict(self, pred_data):
		prob=self.estimator.score_samples(pred_data);
		return -prob