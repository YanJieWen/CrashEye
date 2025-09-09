'''
@File: ult_tracker.py
@Software: PyCharm
@Author: Yanjie Wen(MitSui)
@Institution: CSU&BUCEA
@E-mail: obitowen@csu.edu.cn
@Time: 9月 02, 2025
@HomePage: https://github.com/YanJieWen
'''

import numpy as np


from yolox.masort import linear_assignment
from yolox.masort.detection import Detection
from yolox.masort.kftrack import KalmanTrack
from yolox.masort.embedding import EmbddingComputer
from yolox.masort.cmc import CMCComputer
from yolox.masort.matching_appear import NearstNeighborDistanceMetric
from yolox.masort.matching_iou import _iou_cost,fuse_cost


import cv2
import os

'''
embed_off/cmc_off/aw_off
'''

class ULTtracker(object):
    def __init__(self,args,frame_rate=30):
        self.args = args
        self.buffer_size = int(frame_rate/30.0*args.track_buffer)
        self.max_age = self.buffer_size

        self.n_init = args.min_init
        self.det_thresh = args.track_thresh+0.1
        self.delta_t = args.delta_t
        self.inertia = args.inertia
        self.track_thresh = args.track_thresh

        self._next_id = 1
        self.frame_id = 0
        # self.w_ass_emb = args.w_ass_emb
        # self.alp_ass_emb = args.alp_ass_emb
        # self.aw_parm = args.aw_parm

        #conditions
        self.embed_off = args.embed_off
        self.cmc_off = args.cmc_off
        # self.aw_off = args.aw_off

        #cmc&reid
        self.embedder = EmbddingComputer(args.dataset,args.test_dataset)
        self.cmcer = CMCComputer()

        #appearance_matcher
        self.dist_type = args.dist_type #attn
        self.alpha = args.alpha #memory
        self.w_assoc_emb= args.w_assoc_emb
        self.aw_parm = args.aw_parm
        self.nn_matcher = NearstNeighborDistanceMetric(self.dist_type,self.alpha,self.det_thresh,self.w_assoc_emb,self.aw_parm)


        self.tracks = []


        #Associate methods

    def update(self,output_results,img_info,img_size):
        '''
        Integrates measurement-assisted pseudo-trajectory generation (MPG), measurement-assisted momentum estimation(MME), and a parallel
        3-branch association method (P3A) for motion, appearance, and localization
        Args:
            output_results:torch.tensor-->(x1,y1,x2,y2,conf,)相对输入图像坐标系
            img_info:(h,w,fid,vid,img_file)
            img_size:输入图像分辨率尺寸
        Returns:
        '''
        #step1: rescale bbox and get feat and cmc of A
        img_file_name = img_info[4][0]
        if isinstance(img_file_name,str):
            if self.args.dataset=='mot17':
                if self.args.test_dataset:
                    img_file = os.path.join('./datasets/mot','test',img_file_name)
                else:
                    img_file = os.path.join('./datasets/mot', 'train', img_file_name)
            elif self.args.dataset=='mot20':
                if self.args.test_dataset:
                    img_file = os.path.join('./datasets/MOT20', 'test', img_file_name)
                else:
                    img_file = os.path.join('./datasets/MOT20', 'train', img_file_name)
            elif self.args.dataset=='dance':
                if self.args.test_dataset:
                    img_file = os.path.join('./datasets/dancetrack', 'test', img_file_name)
                else:
                    img_file = os.path.join('./datasets/dancetrack', 'val', img_file_name)

            elif self.args.dataset=='crash':
                if self.args.test_dataset:
                    img_file = os.path.join('./datasets/Crash-Seq', 'test', img_file_name)
                else:
                    img_file = os.path.join('./datasets/Crash-Seq', 'train', img_file_name)
            else:
                raise ValueError("Dataset name is not found")

            frame_id = img_info[2].item()
            video_name = img_info[4][0].split('/')[0]
            tag = f'{video_name}:{frame_id}' #MOT17-01-FRCNN:1
            ori_img = cv2.cvtColor(cv2.imread(img_file), cv2.COLOR_BGR2RGB)
        else:
            tag = f'1:{int(img_info[2])}'
            ori_img = cv2.cvtColor(img_file_name,cv2.COLOR_BGR2RGB)
        self.frame_id += 1

        if output_results.shape[1] == 5 :
            scores = output_results[:,4]
            bboxes = output_results[:,:4]
        else:
            output_results = output_results.cpu().numpy()
            scores = output_results[:, 4] * output_results[:, 5]
            bboxes = output_results[:, :4]
        img_h,img_w = img_info[0],img_info[1]
        scale = min(img_size[0]/float(img_h),img_size[1]/float(img_w))
        bboxes /= scale
        # get reid feat
        feats = np.ones((bboxes.shape[0],1))
        if not self.embed_off and bboxes.shape[0]!=0:
            feats = self.embedder.compute_embedding(ori_img,bboxes,tag)#MXD

        #calculate cmc
        if not self.cmc_off:
            transform = self.cmcer.compute_affine(ori_img,bboxes,tag)
            for tr in self.tracks:
                tr.apply_affine_correction(transform)
        #wrap detection class
        if len(bboxes)>0:
            detections = [Detection(tlbr,s,feat) for (tlbr,s,feat) in zip(bboxes,scores,feats)]
        else:
            detections = []
        # #vis1: draw detection results.high-red/low-blue
        # det_arr = get_det_arrs(detections,self.track_thresh)
        # det_root = f'./masort/{video_name}-det/'
        # os.makedirs(det_root,exist_ok=True)
        # save_det_res(det_arr, ori_img, det_root, frame_id)



        #todo: initial conflict_state
        for i,track in enumerate(self.tracks):
            self.tracks[i].conflict_sign = 0
        #step2: kf for track prediction
        self.predict()
        #step3: tracklets and detections assosiation
        matches,unmatched_tracks,unmatched_detections = self._match(detections)
        #step4: tracklets update and management
        output_results = self._management(detections,matches,unmatched_tracks,unmatched_detections)

        # # vis2: draw tracklets.Tentative-yellow/confirmed-purple/lost-green
        # tr_arr = get_tr_arr(self.tracks)
        # track_root = f'./masort/{video_name}-track/'
        # os.makedirs(track_root, exist_ok=True)
        # save_tr_res(tr_arr, ori_img, track_root, self.frame_id)
        return output_results

    def predict(self):
        to_del = []
        for i,track in enumerate(self.tracks):
            pos = track.predict()[0]
            if np.any(np.isnan(pos)):
                #todo:why nan happen
                to_del.append(i)
        for t in reversed(to_del):
            self.tracks.pop(t)

    def _match(self,detections):
        #step1: Set Decomposition
        confirmed_tracks = [i for i,t in enumerate(self.tracks) if t.is_confirmed]
        unconfirmed_tracks = [i for i,t in enumerate(self.tracks) if not t.is_confirmed]
        high_dets = [j for j,d in enumerate(detections) if d.score>self.track_thresh]
        low_dets = [j for j,d in enumerate(detections) if 0.1<d.score<self.track_thresh]

        #step2: high detections assoiate with confirmed tracks-->union assosiaction

        matches_a,unmatched_tracks_a,unmatched_detections = linear_assignment.union_min_cost_matching(self.tracks,detections,confirmed_tracks,
                                                                                                      high_dets,self.nn_matcher,self.args)


        #step3: unmatched and non-lost tracks associate with low_dets
        umun_tracks = [i for i in unmatched_tracks_a if self.tracks[i].time_since_update==1]
        un_matched_tracks_a = [i for i in unmatched_tracks_a if self.tracks[i].time_since_update!=1]

        matches_b,unmatched_tracks_b,_ = linear_assignment.min_cost_matching(_iou_cost,0.5,self.tracks,detections,umun_tracks,low_dets)

        #step4: unconfirmed tracks associates with rest of high associate
        if self.args.mot20:
            matches_c,unmatched_tracks_c,unmatched_detections = linear_assignment.min_cost_matching(_iou_cost,0.7,self.tracks,detections,
                                                                                                       unconfirmed_tracks,unmatched_detections)
        else:
            matches_c, unmatched_tracks_c, unmatched_detections = linear_assignment.min_cost_matching(fuse_cost, 0.7,
                                                                                                      self.tracks,
                                                                                                      detections,
                                                                                                      unconfirmed_tracks,
                                                                                                      unmatched_detections)

        #step5: merge all indexes
        matches = matches_a + matches_b + matches_c
        unmatched_tracks = list(set(un_matched_tracks_a+unmatched_tracks_b+unmatched_tracks_c))
        return matches,unmatched_tracks,unmatched_detections

    def _management(self,detections,matches,unmatched_tracks,unmatched_detections):
        #step1:for matched
        for track_idx,det_idx in matches:
            tlbr = detections[det_idx].tlbr
            feat = detections[det_idx].feature
            score = detections[det_idx].score
            self.tracks[track_idx].update(tlbr,feat,score)
        #step2: for unmatched_tracks
        for umtr in unmatched_tracks:
            self.tracks[umtr].update(None)
            self.tracks[umtr].mark_removed()
        #step3: for high unmatched detections
        for det_idx in unmatched_detections:
            bbox = detections[det_idx].tlbr
            embs = detections[det_idx].feature
            score = detections[det_idx].score
            # if score<self.det_thresh:
            #     continue
            trk = KalmanTrack(bbox,score,self._next_id,self.n_init,self.max_age,self.delta_t,feat=embs)
            if self.frame_id==1:
                trk.state = 2
            self._next_id += 1
            self.tracks.append(trk)
        #step4: update tracks and features dict
        self.tracks = [t for t in self.tracks if not t.is_removed]
        #only confirmed is used for appear-->include lost feats []
        activate_targets = [t.track_id for t in self.tracks if t.is_confirmed]
        features,targets,scores =[],[],[]
        for track in self.tracks:
            if not track.is_confirmed:
                continue
            features += track.feats
            targets += [track.track_id for _ in track.feats]
            scores += [track.score for _ in track.feats]
            track.feats = [] #when unmatched_tracks Keep the features of the last update
        self.nn_matcher.partial_fit(np.asarray(features),np.asarray(targets),np.asarray(scores),activate_targets)
        #step5: output results
        output_results = [t for t in self.tracks if t.is_confirmed and t.time_since_update < 1]
        #todo: it seems better when ignore the fragmented tracks

        # output_results = [t for t in self.tracks if t.is_confirmed and t.time_since_update<1 and (t.hits_streak>self.n_init or self.frame_id<=self.n_init)]
        return output_results